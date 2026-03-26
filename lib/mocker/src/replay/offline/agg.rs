// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::events::{SimulationEvent, SimulationWorkerStage};
use super::runtime_utils::{
    WorkerCompletionPayload, next_timestamp as choose_next_timestamp, pop_next_concurrency_ready,
    pop_next_trace_ready, pop_ready_worker_completion, push_worker_completion,
};
#[cfg(test)]
use super::state::OfflineWorkerSnapshot;
use super::state::{AggRequestState, OfflineWorkerState};
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::router::OfflineReplayRouter;
#[cfg(test)]
use crate::replay::router::OfflineRouterSnapshot;
use crate::replay::{ReplayRouterMode, TraceCollector};
use crate::scheduler::RouterEventVisibility;
use anyhow::bail;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RouterEvent;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use uuid::Uuid;

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

pub(super) struct AggRuntime {
    now_ms: f64,
    next_worker_idx: usize,
    next_event_seq: u64,
    admission: AdmissionSource,
    requests: HashMap<Uuid, AggRequestState>,
    workers: Vec<OfflineWorkerState>,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    mode: ReplayMode,
    router: Option<OfflineReplayRouter>,
    stats: AggRuntimeStats,
    #[cfg(test)]
    worker_active_requests: Vec<Vec<Uuid>>,
    #[cfg(test)]
    stepped: bool,
}

impl AggRuntime {
    /// Create an aggregated offline runtime seeded from an explicit request queue.
    pub(super) fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(
            args,
            router_config,
            AdmissionSource::Requests(pending),
            num_workers,
            mode,
            router_mode,
        )
    }

    /// Create an aggregated offline runtime whose admissions come from a workload driver.
    pub(super) fn new_workload(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        driver: WorkloadDriver,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<Self> {
        Self::new_with_source(
            args,
            router_config,
            AdmissionSource::Workload(driver),
            num_workers,
            mode,
            router_mode,
        )
    }

    /// Shared constructor for both raw-request and workload-driven admissions.
    fn new_with_source(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        admission: AdmissionSource,
        num_workers: usize,
        mode: ReplayMode,
        router_mode: ReplayRouterMode,
    ) -> anyhow::Result<Self> {
        let args = args.clone().normalized()?;
        let router = match router_mode {
            ReplayRouterMode::RoundRobin => None,
            ReplayRouterMode::KvRouter => {
                Some(OfflineReplayRouter::new(&args, router_config, num_workers)?)
            }
        };
        let capture_kv_events = router.is_some();

        Ok(Self {
            now_ms: 0.0,
            next_worker_idx: 0,
            next_event_seq: 0,
            admission,
            requests: HashMap::new(),
            workers: (0..num_workers)
                .map(|worker_idx| {
                    OfflineWorkerState::new(worker_idx, args.clone(), capture_kv_events)
                })
                .collect(),
            collector: TraceCollector::default(),
            events: BinaryHeap::new(),
            mode,
            router,
            #[cfg(test)]
            stats: AggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: AggRuntimeStats,
            #[cfg(test)]
            worker_active_requests: vec![Vec::new(); num_workers],
            #[cfg(test)]
            stepped: false,
        })
    }

    /// Count all requests currently consuming cluster capacity, including router-queued ones.
    fn cluster_in_flight(&self) -> usize {
        self.workers
            .iter()
            .map(OfflineWorkerState::in_flight)
            .sum::<usize>()
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

    /// Pick the next worker in round-robin order.
    fn next_worker(&mut self) -> usize {
        let worker_idx = self.next_worker_idx;
        self.next_worker_idx = (self.next_worker_idx + 1) % self.workers.len();
        worker_idx
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

    /// Fail fast if a router admission points at a worker that does not exist.
    fn validate_worker_idx(&self, worker_idx: usize) -> anyhow::Result<()> {
        if worker_idx >= self.workers.len() {
            bail!("offline replay selected unknown worker index {worker_idx}");
        }
        Ok(())
    }

    /// Deliver a request to a worker and update the runtime's bookkeeping for that assignment.
    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        worker_idx: usize,
    ) -> anyhow::Result<()> {
        self.validate_worker_idx(worker_idx)?;
        self.workers[worker_idx].receive_request(request);
        self.record_dispatch(uuid, worker_idx);
        #[cfg(test)]
        self.worker_active_requests[worker_idx].push(uuid);
        Ok(())
    }

    /// Submit a request to the router and return an immediate admission when one is available.
    fn submit_to_router(
        &mut self,
        request: &DirectRequest,
        replay_hashes: Option<ReplayRequestHashes>,
    ) -> anyhow::Result<Option<usize>> {
        let Some(router) = self.router.as_mut() else {
            bail!("offline replay router submission requires an active router");
        };
        let maybe_worker_idx =
            router.submit_request_with_hashes(request, replay_hashes, self.now_ms)?;
        self.record_router_pending();
        Ok(maybe_worker_idx)
    }

    /// Materialize router admissions into concrete worker dispatches.
    fn dispatch_router_admissions(&mut self, admissions: Vec<(Uuid, usize)>) -> anyhow::Result<()> {
        for (uuid, worker_idx) in admissions {
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
        if matches!(self.mode, ReplayMode::Concurrency { .. }) {
            request.arrival_timestamp_ms = Some(arrival_time_ms);
        }

        self.collector.on_arrival(
            uuid,
            arrival_time_ms,
            request.tokens.len(),
            request.max_output_tokens,
        );

        if self.router.is_none() {
            self.requests.insert(uuid, AggRequestState::new_running());
            let worker_idx = self.next_worker();
            self.dispatch_to_worker(request, uuid, worker_idx)?;
            return Ok(uuid);
        }
        let maybe_worker_idx = self.submit_to_router(&request, replay_hashes)?;
        if let Some(worker_idx) = maybe_worker_idx {
            self.requests.insert(uuid, AggRequestState::new_running());
            self.dispatch_to_worker(request, uuid, worker_idx)?;
            return Ok(uuid);
        }

        self.requests
            .insert(uuid, AggRequestState::new_queued(request));
        self.record_in_flight_peak();
        Ok(uuid)
    }

    /// Return true once no workers, router queues, or admissions remain.
    fn is_done(&self) -> bool {
        self.events.is_empty()
            && self.cluster_in_flight() == 0
            && match &self.admission {
                AdmissionSource::Requests(pending) => pending.is_empty(),
                AdmissionSource::Workload(driver) => driver.is_drained(),
            }
            && self.workers.iter().all(OfflineWorkerState::is_drained)
    }

    /// Pick the next logical timestamp from either arrivals or scheduled worker completions.
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

    /// Release completed requests from worker-local accounting after a pass finishes.
    fn apply_completed_requests(&mut self, worker_idx: usize, completed_requests: usize) {
        self.workers[worker_idx].mark_completed(completed_requests);
    }

    /// Apply router-visible KV events at the phase chosen by the scheduler core.
    fn apply_router_events(&mut self, events: Vec<RouterEvent>) -> anyhow::Result<()> {
        let Some(router) = self.router.as_mut() else {
            return Ok(());
        };
        for event in events {
            router.apply_event(event)?;
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
                admissions = router.free(signal.uuid)?;
                #[cfg(test)]
                {
                    self.stats.router_freed_count += 1;
                }
                self.record_router_pending();
            }
            self.requests.remove(&signal.uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?;
            if let AdmissionSource::Workload(driver) = &mut self.admission {
                driver.on_complete(signal.uuid, self.now_ms)?;
            }
            self.dispatch_router_admissions(admissions)?;
            return Ok(());
        }

        let already_marked = self
            .requests
            .get(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .prefill_completed();
        if already_marked {
            return Ok(());
        }

        self.requests
            .get_mut(&signal.uuid)
            .ok_or_else(|| {
                anyhow::anyhow!("offline replay missing request state for {}", signal.uuid)
            })?
            .mark_prefill_completed();
        if let Some(router) = self.router.as_mut() {
            admissions = router.mark_prefill_completed(signal.uuid)?;
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
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        self.apply_completed_requests(worker_idx, completed_requests);
        self.apply_router_events(kv_events)?;
        for signal in output_signals {
            self.process_output_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        while let Some(WorkerCompletionPayload {
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            kv_events,
        }) = pop_ready_worker_completion(&mut self.events, self.now_ms)
        {
            debug_assert_eq!(stage, SimulationWorkerStage::Aggregated);
            self.workers[worker_idx].mark_idle();
            self.process_completed_pass(worker_idx, completed_requests, output_signals, kv_events)?;
            changed = true;
        }

        Ok(changed)
    }

    /// Release every trace arrival whose timestamp is now visible to the global clock.
    fn release_trace_arrivals(&mut self) -> anyhow::Result<bool> {
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
                self.assign_request(request, arrival_ms, None)?;
                released_any = true;
            }
            return Ok(released_any);
        }

        let ready_requests = match &mut self.admission {
            AdmissionSource::Requests(_) => unreachable!(),
            AdmissionSource::Workload(driver) => driver.pop_ready(self.now_ms, usize::MAX),
        };
        for ready in ready_requests {
            self.assign_request(
                ready.request,
                ready.scheduled_ready_at_ms,
                ready.replay_hashes,
            )?;
            released_any = true;
        }

        Ok(released_any)
    }

    /// Backfill closed-loop concurrency replay until the configured in-flight limit is reached.
    fn top_off_concurrency(&mut self, max_in_flight: usize) -> anyhow::Result<bool> {
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
                self.assign_request(request, arrival_ms, None)?;
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
            self.assign_request(ready.request, self.now_ms, ready.replay_hashes)?;
            released_any = true;
        }

        Ok(released_any)
    }

    /// Start passes on every idle worker that can make progress at the current timestamp.
    fn drive_ready_workers(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        for worker_idx in 0..self.workers.len() {
            loop {
                if !self.workers[worker_idx].is_ready() {
                    break;
                }

                let executed = {
                    let (workers, collector) = (&mut self.workers, &mut self.collector);
                    workers[worker_idx].execute_pass(collector, self.now_ms)
                };
                changed = true;

                let completion_kv_events =
                    if executed.router_event_visibility == RouterEventVisibility::PassStart {
                        self.apply_router_events(executed.kv_events)?;
                        Vec::new()
                    } else {
                        executed.kv_events
                    };

                if executed.end_ms == self.now_ms {
                    self.process_completed_pass(
                        worker_idx,
                        executed.completed_requests,
                        executed.output_signals,
                        completion_kv_events,
                    )?;
                    continue;
                }

                self.workers[worker_idx].mark_busy();
                push_worker_completion(
                    &mut self.events,
                    &mut self.next_event_seq,
                    executed.end_ms,
                    WorkerCompletionPayload {
                        stage: SimulationWorkerStage::Aggregated,
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

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> anyhow::Result<()> {
        loop {
            let mut changed = self.apply_worker_completions()?;

            changed |= match self.mode {
                ReplayMode::Trace => self.release_trace_arrivals()?,
                ReplayMode::Concurrency { max_in_flight } => {
                    self.top_off_concurrency(max_in_flight)?
                }
            };

            changed |= self.drive_ready_workers()?;

            if !changed {
                break;
            }
        }

        Ok(())
    }

    /// Run the aggregated offline replay until all arrivals and worker work are exhausted.
    pub(super) fn run(mut self) -> anyhow::Result<(TraceCollector, AggRuntimeStats)> {
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
            .filter(|(_, state)| state.is_queued_at_router())
            .map(|(uuid, _)| *uuid)
            .collect::<Vec<_>>();
        router_pending_request_ids.sort_unstable();
        let mut prefill_completed = self
            .requests
            .iter()
            .filter(|(_, state)| state.prefill_completed())
            .map(|(uuid, _)| *uuid)
            .collect::<Vec<_>>();
        prefill_completed.sort_unstable();

        AggRuntimeSnapshot {
            now_ms: self.now_ms,
            worker_active_requests: self.worker_active_requests.clone(),
            workers: self
                .workers
                .iter()
                .map(OfflineWorkerState::debug_snapshot)
                .collect(),
            router_pending_request_ids,
            prefill_completed,
            router: self
                .router
                .as_ref()
                .map(OfflineReplayRouter::debug_snapshot),
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
    use dynamo_kv_router::config::RouterQueuePolicy;

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
}
