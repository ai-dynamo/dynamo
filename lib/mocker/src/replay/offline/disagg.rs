// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, VecDeque};

use anyhow::{Result, anyhow, bail};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RouterEvent;
use uuid::Uuid;

use super::events::{SimulationEvent, SimulationWorkerStage};
use super::runtime_utils::{
    WorkerCompletionPayload, next_timestamp as choose_next_timestamp, pop_next_concurrency_ready,
    pop_next_trace_ready, pop_ready_decode_handoff, pop_ready_worker_completion,
    push_decode_handoff, push_worker_completion,
};
#[cfg(test)]
use super::state::DisaggPhase;
#[cfg(test)]
use super::state::DisaggRequestSnapshot;
use super::state::{DisaggRequestState, OfflineWorkerState};
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::router::OfflineReplayRouter;
use crate::replay::{OfflineDisaggReplayConfig, ReplayRouterMode, TraceCollector};
use crate::scheduler::RouterEventVisibility;

#[derive(Debug, Clone, Copy)]
pub(super) enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct DisaggRuntimeStats {
    request_snapshots: HashMap<Uuid, DisaggRequestSnapshot>,
    prefill_assignments: HashMap<Uuid, usize>,
    decode_assignments: HashMap<Uuid, usize>,
    handoff_ms: HashMap<Uuid, f64>,
    prefill_marked_count: usize,
    prefill_router_freed_count: usize,
    decode_router_freed_count: usize,
    max_prefill_router_pending_count: usize,
    max_decode_router_pending_count: usize,
}

#[cfg(not(test))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct DisaggRuntimeStats;

pub(super) struct DisaggRuntime {
    now_ms: f64,
    next_prefill_worker_idx: usize,
    next_decode_worker_idx: usize,
    next_event_seq: u64,
    admission: AdmissionSource,
    prefill_workers: Vec<OfflineWorkerState>,
    decode_workers: Vec<OfflineWorkerState>,
    prefill_router: Option<OfflineReplayRouter>,
    decode_router: Option<OfflineReplayRouter>,
    requests: HashMap<Uuid, DisaggRequestState>,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    mode: ReplayMode,
    stats: DisaggRuntimeStats,
}

impl DisaggRuntime {
    /// Create a disaggregated offline runtime seeded from an explicit request queue.
    pub(super) fn new(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        pending: VecDeque<DirectRequest>,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        Self::new_with_source(
            config,
            router_config,
            AdmissionSource::Requests(pending),
            mode,
            router_mode,
        )
    }

    /// Create a disaggregated offline runtime whose admissions come from a workload driver.
    pub(super) fn new_workload(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        driver: WorkloadDriver,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        Self::new_with_source(
            config,
            router_config,
            AdmissionSource::Workload(driver),
            mode,
            router_mode,
        )
    }

    /// Shared constructor for both raw-request and workload-driven admissions.
    fn new_with_source(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        admission: AdmissionSource,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
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
                        config.num_prefill_workers,
                    )?),
                    Some(OfflineReplayRouter::new(
                        &config.decode_args,
                        Some(decode_router_config),
                        config.num_decode_workers,
                    )?),
                )
            }
        };

        Ok(Self {
            now_ms: 0.0,
            next_prefill_worker_idx: 0,
            next_decode_worker_idx: 0,
            next_event_seq: 0,
            admission,
            prefill_workers: (0..config.num_prefill_workers)
                .map(|worker_idx| {
                    OfflineWorkerState::new(
                        worker_idx,
                        config.prefill_args.clone(),
                        prefill_router.is_some(),
                    )
                })
                .collect(),
            decode_workers: (0..config.num_decode_workers)
                .map(|worker_idx| {
                    OfflineWorkerState::new(worker_idx, config.decode_args.clone(), false)
                })
                .collect(),
            prefill_router,
            decode_router,
            requests: HashMap::new(),
            collector: TraceCollector::default(),
            events: BinaryHeap::new(),
            mode,
            #[cfg(test)]
            stats: DisaggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: DisaggRuntimeStats,
        })
    }

    /// Count all requests consuming cluster capacity across prefill, decode, and router queues.
    fn cluster_in_flight(&self) -> usize {
        self.prefill_workers
            .iter()
            .map(OfflineWorkerState::in_flight)
            .sum::<usize>()
            + self
                .decode_workers
                .iter()
                .map(OfflineWorkerState::in_flight)
                .sum::<usize>()
            + self
                .prefill_router
                .as_ref()
                .map_or(0, OfflineReplayRouter::pending_count)
            + self
                .decode_router
                .as_ref()
                .map_or(0, OfflineReplayRouter::pending_count)
    }

    /// Pick the next prefill worker in round-robin order.
    fn next_prefill_worker(&mut self) -> usize {
        let worker_idx = self.next_prefill_worker_idx;
        self.next_prefill_worker_idx =
            (self.next_prefill_worker_idx + 1) % self.prefill_workers.len();
        worker_idx
    }

    /// Pick the next decode worker in round-robin order.
    fn next_decode_worker(&mut self) -> usize {
        let worker_idx = self.next_decode_worker_idx;
        self.next_decode_worker_idx = (self.next_decode_worker_idx + 1) % self.decode_workers.len();
        worker_idx
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

    /// Fail fast if a router admission targets a non-existent stage worker.
    fn validate_worker_idx(&self, stage: SimulationWorkerStage, worker_idx: usize) -> Result<()> {
        let worker_count = match stage {
            SimulationWorkerStage::Prefill => self.prefill_workers.len(),
            SimulationWorkerStage::Decode => self.decode_workers.len(),
            SimulationWorkerStage::Aggregated => unreachable!("aggregated stage is not used"),
        };
        if worker_idx >= worker_count {
            bail!("offline disagg replay selected unknown {stage:?} worker index {worker_idx}");
        }
        Ok(())
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

    /// Dispatch a request's prefill stage onto a specific prefill worker.
    fn dispatch_prefill(&mut self, uuid: Uuid, worker_idx: usize) -> Result<()> {
        self.validate_worker_idx(SimulationWorkerStage::Prefill, worker_idx)?;
        let request = self.state(uuid)?.build_prefill_request()?;
        self.prefill_workers[worker_idx].receive_request(request);
        self.state_mut(uuid)?.start_prefill(worker_idx);
        #[cfg(test)]
        {
            self.stats.prefill_assignments.insert(uuid, worker_idx);
        }
        Ok(())
    }

    /// Dispatch a request's decode stage onto a specific decode worker.
    fn dispatch_decode(&mut self, uuid: Uuid, worker_idx: usize) -> Result<()> {
        self.validate_worker_idx(SimulationWorkerStage::Decode, worker_idx)?;
        let request = self.state(uuid)?.build_decode_request()?;
        self.decode_workers[worker_idx].receive_request(request);
        self.state_mut(uuid)?.start_decode(worker_idx);
        #[cfg(test)]
        {
            self.stats.decode_assignments.insert(uuid, worker_idx);
        }
        Ok(())
    }

    /// Turn prefill router admissions into concrete worker dispatches.
    fn dispatch_prefill_admissions(&mut self, admissions: Vec<(Uuid, usize)>) -> Result<()> {
        for (uuid, worker_idx) in admissions {
            if !self.state(uuid)?.is_queued_prefill() {
                bail!("offline disagg replay expected queued prefill request for {uuid}");
            }
            self.dispatch_prefill(uuid, worker_idx)?;
        }
        Ok(())
    }

    /// Turn decode router admissions into concrete worker dispatches.
    fn dispatch_decode_admissions(&mut self, admissions: Vec<(Uuid, usize)>) -> Result<()> {
        for (uuid, worker_idx) in admissions {
            if !self.state(uuid)?.is_queued_decode() {
                bail!("offline disagg replay expected queued decode request for {uuid}");
            }
            self.dispatch_decode(uuid, worker_idx)?;
        }
        Ok(())
    }

    /// Submit a request to the prefill router and return an immediate admission when available.
    fn submit_to_prefill_router(
        &mut self,
        uuid: Uuid,
        replay_hashes: Option<ReplayRequestHashes>,
    ) -> Result<Option<usize>> {
        let request = self.state(uuid)?.original_request()?.clone();
        let Some(prefill_router) = self.prefill_router.as_mut() else {
            bail!("offline disagg replay prefill submission requires an active router");
        };
        let maybe_worker_idx =
            prefill_router.submit_request_with_hashes(&request, replay_hashes, self.now_ms)?;
        self.record_router_pending();
        Ok(maybe_worker_idx)
    }

    /// Submit a request to the decode router and return an immediate admission when available.
    fn submit_to_decode_router(&mut self, uuid: Uuid) -> Result<Option<usize>> {
        let request = self.state(uuid)?.original_request()?.clone();
        let Some(decode_router) = self.decode_router.as_mut() else {
            bail!("offline disagg replay decode submission requires an active router");
        };
        let maybe_worker_idx =
            decode_router.submit_request_with_hashes(&request, None, self.now_ms)?;
        self.record_router_pending();
        Ok(maybe_worker_idx)
    }

    /// Queue or dispatch a request into decode, depending on whether a decode router is active.
    fn enqueue_decode(&mut self, uuid: Uuid) -> Result<()> {
        if self.decode_router.is_none() {
            let worker_idx = self.next_decode_worker();
            self.dispatch_decode(uuid, worker_idx)?;
            return Ok(());
        }
        let maybe_worker_idx = self.submit_to_decode_router(uuid)?;
        #[cfg(test)]
        {
            self.stats.handoff_ms.insert(uuid, self.now_ms);
        }
        if let Some(worker_idx) = maybe_worker_idx {
            self.dispatch_decode(uuid, worker_idx)?;
            return Ok(());
        }

        self.state_mut(uuid)?.queue_decode();
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

        self.requests
            .insert(uuid, DisaggRequestState::new(request, arrival_time_ms));
        if self.prefill_router.is_none() {
            let worker_idx = self.next_prefill_worker();
            self.dispatch_prefill(uuid, worker_idx)?;
            return Ok(uuid);
        }
        let maybe_worker_idx = self.submit_to_prefill_router(uuid, replay_hashes)?;
        if let Some(worker_idx) = maybe_worker_idx {
            self.dispatch_prefill(uuid, worker_idx)?;
        }
        Ok(uuid)
    }

    /// Return true once both stages, both routers, and all admissions are fully drained.
    fn is_done(&self) -> bool {
        self.events.is_empty()
            && self.cluster_in_flight() == 0
            && match &self.admission {
                AdmissionSource::Requests(pending) => pending.is_empty(),
                AdmissionSource::Workload(driver) => driver.is_drained(),
            }
            && self
                .prefill_workers
                .iter()
                .all(OfflineWorkerState::is_drained)
            && self
                .decode_workers
                .iter()
                .all(OfflineWorkerState::is_drained)
    }

    /// Pick the next logical timestamp from arrivals, worker completions, or decode handoffs.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let cluster_in_flight = self.cluster_in_flight();
        let next_arrival_ms = match (&self.mode, &mut self.admission) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Workload(driver)) => {
                if cluster_in_flight < *max_in_flight {
                    driver.next_ready_time_ms()
                } else {
                    None
                }
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
        };

        choose_next_timestamp(next_arrival_ms, next_event_ms)
    }

    /// Apply prefill-side KV router events at the scheduler-selected visibility phase.
    fn apply_prefill_router_events(&mut self, events: Vec<RouterEvent>) -> Result<()> {
        let Some(prefill_router) = self.prefill_router.as_mut() else {
            return Ok(());
        };
        for event in events {
            prefill_router.apply_event(event)?;
        }
        Ok(())
    }

    /// Process one prefill output signal, including router updates and decode handoff scheduling.
    fn process_prefill_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.completed {
            return Ok(());
        }

        if self.prefill_router.is_some() {
            let prefill_complete_admissions = {
                let prefill_router = self.prefill_router.as_mut().expect("router checked above");
                prefill_router.mark_prefill_completed(signal.uuid)?
            };
            #[cfg(test)]
            {
                self.stats.prefill_marked_count += 1;
            }
            self.record_router_pending();
            self.dispatch_prefill_admissions(prefill_complete_admissions)?;

            let admissions = {
                let prefill_router = self.prefill_router.as_mut().expect("router checked above");
                prefill_router.free(signal.uuid)?
            };
            #[cfg(test)]
            {
                self.stats.prefill_router_freed_count += 1;
            }
            self.record_router_pending();
            self.dispatch_prefill_admissions(admissions)?;
        }

        self.enqueue_decode_after_handoff(signal.uuid, signal.handoff_delay_ms)
    }

    /// Process one decode output signal, including decode router frees and request completion.
    fn process_decode_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.completed {
            return Ok(());
        }

        let admissions = if let Some(decode_router) = self.decode_router.as_mut() {
            let admissions = decode_router.free(signal.uuid)?;
            #[cfg(test)]
            {
                self.stats.decode_router_freed_count += 1;
            }
            admissions
        } else {
            Vec::new()
        };
        self.record_router_pending();
        if let AdmissionSource::Workload(driver) = &mut self.admission {
            driver.on_complete(signal.uuid, self.now_ms)?;
        }
        self.state_mut(signal.uuid)?.mark_done();
        self.dispatch_decode_admissions(admissions)?;
        Ok(())
    }

    /// Apply the side effects of a finished prefill pass.
    fn process_prefill_pass(
        &mut self,
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<RouterEvent>,
    ) -> Result<()> {
        self.prefill_workers[worker_idx].mark_completed(completed_requests);
        self.apply_prefill_router_events(kv_events)?;
        for signal in output_signals {
            self.process_prefill_signal(signal)?;
        }
        Ok(())
    }

    /// Apply the side effects of a finished decode pass.
    fn process_decode_pass(
        &mut self,
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
    ) -> Result<()> {
        self.decode_workers[worker_idx].mark_completed(completed_requests);
        for signal in output_signals {
            self.process_decode_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some(WorkerCompletionPayload {
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            kv_events,
        }) = pop_ready_worker_completion(&mut self.events, self.now_ms)
        {
            match stage {
                SimulationWorkerStage::Prefill => {
                    self.prefill_workers[worker_idx].mark_idle();
                    self.process_prefill_pass(
                        worker_idx,
                        completed_requests,
                        output_signals,
                        kv_events,
                    )?;
                }
                SimulationWorkerStage::Decode => {
                    self.decode_workers[worker_idx].mark_idle();
                    self.process_decode_pass(worker_idx, completed_requests, output_signals)?;
                }
                SimulationWorkerStage::Aggregated => {
                    bail!("offline disagg replay received an aggregated completion event")
                }
            }
            changed = true;
        }
        Ok(changed)
    }

    /// Drain all delayed decode handoff events scheduled for the current logical timestamp.
    fn apply_decode_handoffs(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some(uuid) = pop_ready_decode_handoff(&mut self.events, self.now_ms) {
            self.enqueue_decode(uuid)?;
            changed = true;
        }
        Ok(changed)
    }

    /// Either enqueue decode immediately or schedule a delayed handoff event on the event heap.
    fn enqueue_decode_after_handoff(
        &mut self,
        uuid: Uuid,
        handoff_delay_ms: Option<f64>,
    ) -> Result<()> {
        if let Some(delay_ms) = handoff_delay_ms
            && delay_ms > 0.0
        {
            push_decode_handoff(
                &mut self.events,
                &mut self.next_event_seq,
                self.now_ms + delay_ms,
                uuid,
            );
            return Ok(());
        }

        self.enqueue_decode(uuid)
    }

    /// Release every trace arrival whose timestamp is now visible to the global clock.
    fn release_trace_arrivals(&mut self) -> Result<bool> {
        let mut released_any = false;
        if matches!(self.admission, AdmissionSource::Requests(_)) {
            loop {
                let next_ready = match &mut self.admission {
                    AdmissionSource::Requests(pending) => {
                        pop_next_trace_ready(pending, self.now_ms)
                    }
                    AdmissionSource::Workload(_) => unreachable!(),
                };
                let Some((request, arrival_ms)) = next_ready else {
                    break;
                };
                self.on_external_arrival(request, arrival_ms, None)?;
                released_any = true;
            }
            return Ok(released_any);
        }

        let ready_requests = match &mut self.admission {
            AdmissionSource::Requests(_) => unreachable!(),
            AdmissionSource::Workload(driver) => driver.pop_ready(self.now_ms, usize::MAX),
        };
        for ready in ready_requests {
            self.on_external_arrival(
                ready.request,
                ready.scheduled_ready_at_ms,
                ready.replay_hashes,
            )?;
            released_any = true;
        }
        Ok(released_any)
    }

    /// Backfill closed-loop concurrency replay until the staged cluster reaches its cap.
    fn top_off_concurrency(&mut self, max_in_flight: usize) -> Result<bool> {
        let mut released_any = false;
        if matches!(self.admission, AdmissionSource::Requests(_)) {
            loop {
                let cluster_in_flight = self.cluster_in_flight();
                let next_ready = match &mut self.admission {
                    AdmissionSource::Requests(pending) => pop_next_concurrency_ready(
                        pending,
                        self.now_ms,
                        cluster_in_flight,
                        max_in_flight,
                    ),
                    AdmissionSource::Workload(_) => unreachable!(),
                };
                let Some((request, arrival_ms)) = next_ready else {
                    break;
                };
                self.on_external_arrival(request, arrival_ms, None)?;
                released_any = true;
            }
            return Ok(released_any);
        }

        let available = max_in_flight.saturating_sub(self.cluster_in_flight());
        if available == 0 {
            return Ok(false);
        }

        let ready_requests = match &mut self.admission {
            AdmissionSource::Requests(_) => unreachable!(),
            AdmissionSource::Workload(driver) => driver.pop_ready(self.now_ms, available),
        };
        for ready in ready_requests {
            self.on_external_arrival(ready.request, self.now_ms, ready.replay_hashes)?;
            released_any = true;
        }
        Ok(released_any)
    }

    /// Start passes on every idle prefill worker that can make progress at the current timestamp.
    fn drive_prefill_workers(&mut self) -> Result<bool> {
        let mut changed = false;
        for worker_idx in 0..self.prefill_workers.len() {
            loop {
                if !self.prefill_workers[worker_idx].is_ready() {
                    break;
                }

                let executed = self.prefill_workers[worker_idx].execute_hidden_pass(self.now_ms);
                changed = true;

                let completion_kv_events =
                    if executed.router_event_visibility == RouterEventVisibility::PassStart {
                        self.apply_prefill_router_events(executed.kv_events)?;
                        Vec::new()
                    } else {
                        executed.kv_events
                    };

                if executed.end_ms == self.now_ms {
                    self.process_prefill_pass(
                        worker_idx,
                        executed.completed_requests,
                        executed.output_signals,
                        completion_kv_events,
                    )?;
                    continue;
                }

                self.prefill_workers[worker_idx].mark_busy();
                push_worker_completion(
                    &mut self.events,
                    &mut self.next_event_seq,
                    executed.end_ms,
                    WorkerCompletionPayload {
                        stage: SimulationWorkerStage::Prefill,
                        worker_idx,
                        completed_requests: executed.completed_requests,
                        output_signals: executed.output_signals,
                        kv_events: completion_kv_events,
                    },
                );
                break;
            }
        }
        Ok(changed)
    }

    /// Start passes on every idle decode worker that can make progress at the current timestamp.
    fn drive_decode_workers(&mut self) -> Result<bool> {
        let mut changed = false;
        for worker_idx in 0..self.decode_workers.len() {
            loop {
                if !self.decode_workers[worker_idx].is_ready() {
                    break;
                }

                let executed = {
                    let (workers, collector) = (&mut self.decode_workers, &mut self.collector);
                    workers[worker_idx].execute_pass(collector, self.now_ms)
                };
                changed = true;

                if executed.end_ms == self.now_ms {
                    self.process_decode_pass(
                        worker_idx,
                        executed.completed_requests,
                        executed.output_signals,
                    )?;
                    continue;
                }

                self.decode_workers[worker_idx].mark_busy();
                push_worker_completion(
                    &mut self.events,
                    &mut self.next_event_seq,
                    executed.end_ms,
                    WorkerCompletionPayload {
                        stage: SimulationWorkerStage::Decode,
                        worker_idx,
                        completed_requests: executed.completed_requests,
                        output_signals: executed.output_signals,
                        kv_events: Vec::new(),
                    },
                );
                break;
            }
        }
        Ok(changed)
    }

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> Result<()> {
        loop {
            let mut changed = self.apply_worker_completions()?;
            changed |= self.apply_decode_handoffs()?;

            changed |= match self.mode {
                ReplayMode::Trace => self.release_trace_arrivals()?,
                ReplayMode::Concurrency { max_in_flight } => {
                    self.top_off_concurrency(max_in_flight)?
                }
            };

            changed |= self.drive_prefill_workers()?;
            changed |= self.drive_decode_workers()?;

            if !changed {
                break;
            }
        }
        Ok(())
    }

    /// Finalize test-only request snapshots before returning.
    fn finish_test_stats(&mut self) {
        #[cfg(test)]
        {
            self.stats.request_snapshots = self
                .requests
                .iter()
                .map(|(uuid, state)| (*uuid, state.debug_snapshot()))
                .collect();
        }
    }

    /// Run the staged offline replay until both prefill and decode pipelines are drained.
    pub(super) fn run(mut self) -> Result<(TraceCollector, DisaggRuntimeStats)> {
        self.drain_current_timestamp()?;

        while !self.is_done() {
            let Some(next_timestamp_ms) = self.next_timestamp() else {
                bail!(
                    "offline disagg replay reached a dead end with {} in-flight requests remaining",
                    self.cluster_in_flight()
                );
            };
            self.now_ms = next_timestamp_ms;
            self.drain_current_timestamp()?;
        }

        self.finish_test_stats();
        Ok((self.collector, self.stats))
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
    config.overlap_score_weight = 0.0;
    config.router_assume_kv_reuse = false;
    config.router_track_prefill_tokens = false;
    config
}

#[cfg(test)]
mod tests {
    use super::super::entrypoints::{run_concurrency_collect, run_trace_collect};
    use super::*;
    use crate::common::protocols::{MockEngineArgs, WorkerType};

    fn staged_args(worker_type: WorkerType, speedup_ratio: f64) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(64)
            .num_gpu_blocks(256)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(speedup_ratio)
            .decode_speedup_ratio(speedup_ratio)
            .worker_type(worker_type)
            .build()
            .unwrap()
    }

    fn disagg_config() -> OfflineDisaggReplayConfig {
        OfflineDisaggReplayConfig {
            prefill_args: staged_args(WorkerType::Prefill, 1000.0),
            decode_args: staged_args(WorkerType::Decode, 1000.0),
            num_prefill_workers: 2,
            num_decode_workers: 2,
        }
    }

    fn disagg_config_with_handoff_delay() -> OfflineDisaggReplayConfig {
        let mut config = disagg_config();
        config.prefill_args.kv_transfer_bandwidth = Some(1.0);
        config.prefill_args.kv_bytes_per_token = Some(1_000_000);
        config
    }

    fn router_config() -> KvRouterConfig {
        KvRouterConfig {
            router_queue_threshold: Some(1.25),
            ..KvRouterConfig::default()
        }
    }

    fn request(
        uuid: u128,
        prompt_tokens: usize,
        output_tokens: usize,
        arrival_ms: f64,
    ) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; prompt_tokens],
            max_output_tokens: output_tokens,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(arrival_ms),
        }
    }

    #[test]
    fn test_derive_stage_router_configs_force_required_overrides() {
        let config = KvRouterConfig {
            overlap_score_weight: 2.0,
            router_track_active_blocks: true,
            router_assume_kv_reuse: true,
            router_track_prefill_tokens: true,
            ..KvRouterConfig::default()
        };
        let args = staged_args(WorkerType::Prefill, 1.0);
        let prefill = derive_prefill_router_config(&args, Some(config.clone()));
        let decode = derive_decode_router_config(&args, Some(config));

        assert!(!prefill.router_track_active_blocks);
        assert_eq!(decode.overlap_score_weight, 0.0);
        assert!(!decode.router_assume_kv_reuse);
        assert!(!decode.router_track_prefill_tokens);
    }

    #[rstest::rstest]
    #[case(ReplayRouterMode::RoundRobin)]
    #[case(ReplayRouterMode::KvRouter)]
    fn test_trace_smoke_reports_decode_only_tokens(#[case] router_mode: ReplayRouterMode) {
        let config = disagg_config();
        let requests = vec![request(1, 128, 3, 5.0)];

        let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
        let (collector, stats) =
            run_trace_collect(&config, requests, router_config, 1.0, router_mode);
        let snapshot = collector.snapshot(Uuid::from_u128(1)).unwrap();
        let report = collector.finish();

        assert_eq!(snapshot.arrival_time_ms, 0.0);
        assert!(snapshot.first_admit_ms.is_some());
        assert!(snapshot.first_token_ms.is_some());
        assert_eq!(snapshot.output_length, 3);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(
            stats.request_snapshots[&Uuid::from_u128(1)].phase,
            DisaggPhase::Done
        );
    }

    #[rstest::rstest]
    #[case(ReplayRouterMode::RoundRobin)]
    #[case(ReplayRouterMode::KvRouter)]
    fn test_prefill_and_decode_use_separate_worker_pools(#[case] router_mode: ReplayRouterMode) {
        let config = disagg_config();
        let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 10.0)];

        let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
        let (_, stats) = run_trace_collect(&config, requests, router_config, 1.0, router_mode);

        for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            assert!(stats.prefill_assignments.contains_key(&uuid));
            assert!(stats.decode_assignments.contains_key(&uuid));
        }
    }

    #[test]
    fn test_prefill_overlap_prefers_same_worker_after_handoff_delay() {
        let config = disagg_config();
        let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)];

        let (_, stats) = run_trace_collect(
            &config,
            requests,
            Some(router_config()),
            1.0,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(
            stats.prefill_assignments[&Uuid::from_u128(1)],
            stats.prefill_assignments[&Uuid::from_u128(2)],
        );
    }

    #[rstest::rstest]
    #[case(ReplayRouterMode::RoundRobin)]
    #[case(ReplayRouterMode::KvRouter)]
    fn test_concurrency_backfill_waits_for_decode_completion(
        #[case] router_mode: ReplayRouterMode,
    ) {
        let config = disagg_config();
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 128],
                max_output_tokens: 3,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            },
            DirectRequest {
                tokens: vec![2; 128],
                max_output_tokens: 3,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
            },
        ];

        let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
        let (collector, _) =
            run_concurrency_collect(&config, requests, router_config, 1, router_mode);
        let first = collector.snapshot(Uuid::from_u128(1)).unwrap();
        let second = collector.snapshot(Uuid::from_u128(2)).unwrap();

        assert_eq!(first.arrival_time_ms, 0.0);
        assert_eq!(second.arrival_time_ms, first.last_token_ms.unwrap());
    }

    #[test]
    fn test_prefill_completion_marks_and_frees_before_decode_handoff() {
        let config = disagg_config();
        let requests = vec![request(1, 128, 2, 0.0)];

        let (_, stats) = run_trace_collect(
            &config,
            requests,
            Some(router_config()),
            1.0,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(stats.prefill_marked_count, 1);
        assert_eq!(stats.prefill_router_freed_count, 1);
        assert_eq!(stats.decode_router_freed_count, 1);
    }

    #[test]
    fn test_handoff_delay_increases_decode_visible_ttft() {
        let requests = vec![request(1, 128, 2, 0.0)];

        let (baseline_collector, _) = run_trace_collect(
            &disagg_config(),
            requests.clone(),
            None,
            1.0,
            ReplayRouterMode::RoundRobin,
        );
        let (delayed_collector, _) = run_trace_collect(
            &disagg_config_with_handoff_delay(),
            requests,
            None,
            1.0,
            ReplayRouterMode::RoundRobin,
        );

        let baseline = baseline_collector.snapshot(Uuid::from_u128(1)).unwrap();
        let delayed = delayed_collector.snapshot(Uuid::from_u128(1)).unwrap();
        let baseline_ttft = baseline.first_token_ms.unwrap() - baseline.arrival_time_ms;
        let delayed_ttft = delayed.first_token_ms.unwrap() - delayed.arrival_time_ms;

        assert!(
            delayed_ttft >= baseline_ttft + 120.0,
            "expected delayed TTFT to include roughly 128ms of handoff delay, baseline={baseline_ttft}, delayed={delayed_ttft}"
        );
    }
}
