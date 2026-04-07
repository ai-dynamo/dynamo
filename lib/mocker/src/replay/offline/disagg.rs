// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, VecDeque};

use anyhow::{Result, anyhow, bail};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RouterEvent;
use uuid::Uuid;

pub(super) use super::components::ReplayMode;
use super::components::{
    AdmissionQueue, EngineComponent, EngineEffects, EnginePassMode, OfflineReplayRouter,
    ScheduledWorkerCompletion, WorkerAdmission,
};
use super::events::{SimulationEvent, SimulationWorkerStage};
use super::progress::ReplayProgress;
use super::runtime_utils::{
    next_timestamp as choose_next_timestamp, pop_ready_decode_handoff, pop_ready_worker_completion,
    push_decode_handoff, push_worker_completion,
};
#[cfg(test)]
use super::state::DisaggRequestSnapshot;
use super::state::{DisaggPhase, DisaggRequestState};
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::loadgen::{ReplayRequestHashes, WorkloadDriver};
use crate::replay::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, TraceCollector,
};
use crate::scheduler::AdmissionEvent;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisaggTransition {
    PrefillMarkCompleted { uuid: Uuid },
    PrefillFree { uuid: Uuid },
    DecodeHandoffQueued { uuid: Uuid },
    DecodeEnqueued { uuid: Uuid },
    DecodeFree { uuid: Uuid },
    RequestMarkedDone { uuid: Uuid },
    WorkloadCompleted { uuid: Uuid },
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
    transition_log: Vec<DisaggTransition>,
}

#[cfg(not(test))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct DisaggRuntimeStats;

pub(super) struct DisaggRuntime {
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
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    progress: ReplayProgress,
    stats: DisaggRuntimeStats,
}

impl DisaggRuntime {
    /// Create a disaggregated offline runtime seeded from an explicit request queue.
    pub(super) fn new(
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
        )
    }

    /// Create a disaggregated offline runtime whose admissions come from a workload driver.
    pub(super) fn new_workload(
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
        )
    }

    /// Shared constructor for both raw-request and workload-driven admissions.
    fn new_with_source(
        config: &OfflineDisaggReplayConfig,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        admission: AdmissionQueue,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
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

        let prefill_engine = EngineComponent::new(
            SimulationWorkerStage::Prefill,
            EnginePassMode::Hidden,
            (0..config.num_prefill_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.prefill_args.clone(),
                        prefill_router.is_some(),
                    )
                })
                .collect(),
        );
        let decode_engine = EngineComponent::new(
            SimulationWorkerStage::Decode,
            EnginePassMode::Visible,
            (0..config.num_decode_workers)
                .map(|worker_idx| {
                    super::state::OfflineWorkerState::new(
                        worker_idx,
                        config.decode_args.clone(),
                        false,
                    )
                })
                .collect(),
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
            collector: TraceCollector::default(),
            events: BinaryHeap::new(),
            progress,
            #[cfg(test)]
            stats: DisaggRuntimeStats::default(),
            #[cfg(not(test))]
            stats: DisaggRuntimeStats,
        })
    }

    /// Count all requests consuming cluster capacity across prefill, decode, and router queues.
    fn cluster_in_flight(&self) -> usize {
        self.prefill_engine.in_flight()
            + self.decode_engine.in_flight()
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
            (self.next_prefill_worker_idx + 1) % self.prefill_engine.worker_count();
        worker_idx
    }

    /// Pick the next decode worker in round-robin order.
    fn next_decode_worker(&mut self) -> usize {
        let worker_idx = self.next_decode_worker_idx;
        self.next_decode_worker_idx =
            (self.next_decode_worker_idx + 1) % self.decode_engine.worker_count();
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
        let request = self.state(uuid)?.build_prefill_request()?;
        self.prefill_engine.dispatch(worker_idx, request)?;
        self.state_mut(uuid)?.start_prefill(worker_idx);
        #[cfg(test)]
        {
            self.stats.prefill_assignments.insert(uuid, worker_idx);
        }
        Ok(())
    }

    /// Dispatch a request's decode stage onto a specific decode worker.
    fn dispatch_decode(&mut self, uuid: Uuid, worker_idx: usize) -> Result<()> {
        let request = self.state(uuid)?.original_request()?.clone();
        self.decode_engine.dispatch(worker_idx, request)?;
        self.state_mut(uuid)?.start_decode(worker_idx);
        #[cfg(test)]
        {
            self.stats.decode_assignments.insert(uuid, worker_idx);
        }
        Ok(())
    }

    /// Turn prefill router admissions into concrete worker dispatches.
    fn dispatch_prefill_admissions(&mut self, admissions: Vec<WorkerAdmission>) -> Result<()> {
        for WorkerAdmission { uuid, worker_idx } in admissions {
            if self.state(uuid)?.phase != DisaggPhase::QueuedPrefill {
                bail!("offline disagg replay expected queued prefill request for {uuid}");
            }
            self.dispatch_prefill(uuid, worker_idx)?;
        }
        Ok(())
    }

    /// Turn decode router admissions into concrete worker dispatches.
    fn dispatch_decode_admissions(&mut self, admissions: Vec<WorkerAdmission>) -> Result<()> {
        for WorkerAdmission { uuid, worker_idx } in admissions {
            if self.state(uuid)?.phase != DisaggPhase::QueuedDecode {
                bail!("offline disagg replay expected queued decode request for {uuid}");
            }
            self.dispatch_decode(uuid, worker_idx)?;
        }
        Ok(())
    }

    /// Queue or dispatch a request into decode, depending on whether a decode router is active.
    fn enqueue_decode(&mut self, uuid: Uuid) -> Result<()> {
        if self.decode_router.is_none() {
            #[cfg(test)]
            {
                self.stats
                    .transition_log
                    .push(DisaggTransition::DecodeEnqueued { uuid });
                self.stats.handoff_ms.insert(uuid, self.now_ms);
            }
            let worker_idx = self.next_decode_worker();
            self.dispatch_decode(uuid, worker_idx)?;
            return Ok(());
        }
        let request = self.state(uuid)?.original_request()?.clone();
        self.state_mut(uuid)?.queue_decode();
        #[cfg(test)]
        {
            self.stats
                .transition_log
                .push(DisaggTransition::DecodeEnqueued { uuid });
            self.stats.handoff_ms.insert(uuid, self.now_ms);
        }
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

        let queued_request = request.clone();
        self.requests
            .insert(uuid, DisaggRequestState::new(request, arrival_time_ms));
        if self.prefill_router.is_none() {
            let worker_idx = self.next_prefill_worker();
            self.dispatch_prefill(uuid, worker_idx)?;
            return Ok(uuid);
        }
        let admissions = self
            .prefill_router
            .as_mut()
            .expect("prefill router presence checked above")
            .on_request_arrival(&queued_request, replay_hashes, self.now_ms)?
            .admissions;
        self.record_router_pending();
        self.dispatch_prefill_admissions(admissions)?;
        Ok(uuid)
    }

    /// Return true once both stages, both routers, and all admissions are fully drained.
    fn is_done(&self) -> bool {
        self.events.is_empty()
            && self.cluster_in_flight() == 0
            && self.admission.is_drained()
            && self.prefill_engine.is_drained()
            && self.decode_engine.is_drained()
    }

    /// Pick the next logical timestamp from arrivals, worker completions, or decode handoffs.
    fn next_timestamp(&mut self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        choose_next_timestamp(
            self.admission.next_ready_time_ms(self.cluster_in_flight()),
            next_event_ms,
        )
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

    /// Process one prefill output signal, including router updates and decode handoff scheduling.
    fn process_prefill_signal(&mut self, signal: OutputSignal) -> Result<()> {
        if !signal.completed {
            return Ok(());
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

            let admissions = {
                let prefill_router = self.prefill_router.as_mut().expect("router checked above");
                prefill_router
                    .on_request_completed(signal.uuid, self.now_ms)?
                    .admissions
            };
            #[cfg(test)]
            {
                self.stats.prefill_router_freed_count += 1;
                self.stats
                    .transition_log
                    .push(DisaggTransition::PrefillFree { uuid: signal.uuid });
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
            let admissions = decode_router
                .on_request_completed(signal.uuid, self.now_ms)?
                .admissions;
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
        self.admission
            .on_request_completed(signal.uuid, self.now_ms)?;
        self.progress.inc_completed();
        #[cfg(test)]
        if self.admission.is_workload() {
            self.stats
                .transition_log
                .push(DisaggTransition::WorkloadCompleted { uuid: signal.uuid });
        }
        self.state_mut(signal.uuid)?.mark_done();
        #[cfg(test)]
        {
            self.stats
                .transition_log
                .push(DisaggTransition::RequestMarkedDone { uuid: signal.uuid });
        }
        self.dispatch_decode_admissions(admissions)?;
        Ok(())
    }

    /// Apply the side effects of a finished prefill pass.
    fn process_prefill_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<RouterEvent>,
    ) -> Result<()> {
        self.apply_prefill_router_events(kv_events)?;
        for signal in output_signals {
            self.process_prefill_signal(signal)?;
        }
        Ok(())
    }

    /// Apply the side effects of a finished decode pass.
    fn process_decode_pass(
        &mut self,
        _worker_idx: usize,
        _completed_requests: usize,
        output_signals: Vec<OutputSignal>,
    ) -> Result<()> {
        for signal in output_signals {
            self.process_decode_signal(signal)?;
        }
        Ok(())
    }

    /// Drain all worker-completion events scheduled for the current logical timestamp.
    fn apply_worker_completions(&mut self) -> Result<bool> {
        let mut changed = false;
        while let Some(payload) = pop_ready_worker_completion(&mut self.events, self.now_ms) {
            match payload.stage {
                SimulationWorkerStage::Prefill => {
                    let payload = self.prefill_engine.on_scheduled_completion(payload)?;
                    self.process_prefill_pass(
                        payload.worker_idx,
                        payload.completed_requests,
                        payload.output_signals,
                        payload.kv_events,
                    )?;
                }
                SimulationWorkerStage::Decode => {
                    let payload = self.decode_engine.on_scheduled_completion(payload)?;
                    self.process_decode_pass(
                        payload.worker_idx,
                        payload.completed_requests,
                        payload.output_signals,
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
        let Some(delay_ms) = handoff_delay_ms else {
            return self.enqueue_decode(uuid);
        };
        if delay_ms > 0.0 {
            push_decode_handoff(
                &mut self.events,
                &mut self.next_event_seq,
                self.now_ms + delay_ms,
                uuid,
            );
            #[cfg(test)]
            self.stats
                .transition_log
                .push(DisaggTransition::DecodeHandoffQueued { uuid });
            return Ok(());
        }
        self.enqueue_decode(uuid)
    }

    /// Release every admission made ready by the shared admission queue.
    fn release_ready_arrivals(&mut self) -> Result<bool> {
        let mut released_any = false;
        for ready in self
            .admission
            .drain_ready(self.now_ms, self.cluster_in_flight())?
        {
            self.on_external_arrival(ready.request, ready.arrival_time_ms, ready.replay_hashes)?;
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
        self.record_prefill_admissions(effects.admissions);
        self.apply_prefill_router_events(effects.pass_start_kv_events)?;
        for payload in effects.immediate_completions {
            let payload = self.prefill_engine.on_scheduled_completion(payload)?;
            self.process_prefill_pass(
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

    fn record_prefill_admissions(&mut self, admissions: Vec<AdmissionEvent>) {
        for admission in admissions {
            self.collector
                .on_admit(admission.uuid, self.now_ms, admission.reused_input_tokens);
        }
    }

    fn handle_decode_engine_effects(&mut self, effects: EngineEffects) -> Result<()> {
        for payload in effects.immediate_completions {
            let payload = self.decode_engine.on_scheduled_completion(payload)?;
            self.process_decode_pass(
                payload.worker_idx,
                payload.completed_requests,
                payload.output_signals,
            )?;
        }
        for ScheduledWorkerCompletion { at_ms, payload } in effects.scheduled_completions {
            push_worker_completion(&mut self.events, &mut self.next_event_seq, at_ms, payload);
        }
        Ok(())
    }

    /// Repeatedly process all work that becomes possible without advancing logical time.
    fn drain_current_timestamp(&mut self) -> Result<()> {
        loop {
            let mut changed = self.apply_worker_completions()?;
            changed |= self.apply_decode_handoffs()?;
            changed |= self.release_ready_arrivals()?;
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

        self.progress.finish();
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
    config.router_prefill_load_model = dynamo_kv_router::config::RouterPrefillLoadModel::None;
    config
}

#[cfg(test)]
mod tests {
    use super::super::entrypoints::{
        run_concurrency_collect, run_concurrency_workload_collect, run_trace_collect,
        run_trace_workload_collect,
    };
    use super::*;
    use crate::common::protocols::{EngineType, MockEngineArgs, SglangArgs, WorkerType};
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};

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

    fn sglang_staged_args(worker_type: WorkerType, speedup_ratio: f64) -> MockEngineArgs {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .block_size(64)
            .num_gpu_blocks(512)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(8))
            .enable_prefix_caching(true)
            .enable_chunked_prefill(true)
            .speedup_ratio(speedup_ratio)
            .decode_speedup_ratio(speedup_ratio)
            .worker_type(worker_type)
            .sglang(Some(SglangArgs {
                page_size: Some(64),
                ..Default::default()
            }))
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

    fn sglang_disagg_config() -> OfflineDisaggReplayConfig {
        OfflineDisaggReplayConfig {
            prefill_args: sglang_staged_args(WorkerType::Prefill, 1000.0),
            decode_args: sglang_staged_args(WorkerType::Decode, 1000.0),
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

    fn transition_index(transitions: &[DisaggTransition], needle: DisaggTransition) -> usize {
        transitions
            .iter()
            .position(|transition| *transition == needle)
            .unwrap()
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
        assert_eq!(report.request_counts.total_output_tokens, 3);
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
            assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
            assert_eq!(
                stats.request_snapshots[&uuid].prefill_worker_idx,
                Some(stats.prefill_assignments[&uuid])
            );
            assert_eq!(
                stats.request_snapshots[&uuid].decode_worker_idx,
                Some(stats.decode_assignments[&uuid])
            );
        }
    }

    #[test]
    fn test_prefill_overlap_prefers_same_worker_after_handoff_delay() {
        let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)];

        let cases = [(disagg_config(), true), (sglang_disagg_config(), false)];
        for (config, expect_same_worker) in cases {
            let (_, stats) = run_trace_collect(
                &config,
                requests.clone(),
                Some(router_config()),
                1.0,
                ReplayRouterMode::KvRouter,
            );

            if expect_same_worker {
                assert_eq!(
                    stats.prefill_assignments[&Uuid::from_u128(1)],
                    stats.prefill_assignments[&Uuid::from_u128(2)],
                );
            } else {
                for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
                    assert!(stats.prefill_assignments.contains_key(&uuid));
                    assert!(stats.decode_assignments.contains_key(&uuid));
                    assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
                }
            }
        }
    }

    #[test]
    fn test_hidden_prefill_reports_reused_tokens_even_when_decode_prefix_caching_is_disabled() {
        let mut config = disagg_config();
        config.num_prefill_workers = 1;
        config.num_decode_workers = 1;
        config.decode_args.enable_prefix_caching = false;

        let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)];
        let (collector, _) = run_trace_collect(
            &config,
            requests,
            Some(router_config()),
            1.0,
            ReplayRouterMode::KvRouter,
        );

        let request_2 = collector.snapshot(Uuid::from_u128(2)).unwrap();
        let report = collector.finish();

        assert!(request_2.reused_input_tokens > 0);
        assert!(report.prefix_cache_reused_ratio > 0.0);
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
        let (collector, stats) =
            run_concurrency_collect(&config, requests, router_config, 1, router_mode);
        let first = collector.snapshot(Uuid::from_u128(1)).unwrap();
        let second = collector.snapshot(Uuid::from_u128(2)).unwrap();

        assert_eq!(first.arrival_time_ms, 0.0);
        assert_eq!(second.arrival_time_ms, first.last_token_ms.unwrap());
        assert_eq!(
            stats.request_snapshots[&Uuid::from_u128(1)].phase,
            DisaggPhase::Done
        );
        assert_eq!(
            stats.request_snapshots[&Uuid::from_u128(2)].phase,
            DisaggPhase::Done
        );
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
        let transitions = &stats.transition_log;
        let uuid = Uuid::from_u128(1);
        let mark_idx =
            transition_index(transitions, DisaggTransition::PrefillMarkCompleted { uuid });
        let free_idx = transition_index(transitions, DisaggTransition::PrefillFree { uuid });
        let enqueue_idx = transition_index(transitions, DisaggTransition::DecodeEnqueued { uuid });
        assert!(mark_idx < free_idx);
        assert!(free_idx < enqueue_idx);
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
        let (delayed_collector, delayed_stats) = run_trace_collect(
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
        let uuid = Uuid::from_u128(1);
        let queued_idx = transition_index(
            &delayed_stats.transition_log,
            DisaggTransition::DecodeHandoffQueued { uuid },
        );
        let enqueued_idx = transition_index(
            &delayed_stats.transition_log,
            DisaggTransition::DecodeEnqueued { uuid },
        );
        assert!(queued_idx < enqueued_idx);
        assert!(delayed_stats.handoff_ms[&uuid] >= 120.0);
    }

    #[test]
    fn test_trace_workload_follow_up_turn_arrives_after_completion_plus_delay() {
        let (collector, _) = run_trace_workload_collect(
            &disagg_config(),
            multiturn_trace(),
            None,
            ReplayRouterMode::RoundRobin,
        );
        let snapshots = collector.snapshots();
        let first_turn = snapshots
            .iter()
            .find(|snapshot| snapshot.input_length == 64)
            .unwrap();
        let second_turn = snapshots
            .iter()
            .find(|snapshot| snapshot.input_length == 192)
            .unwrap();
        let session_b = snapshots
            .iter()
            .find(|snapshot| snapshot.input_length == 128)
            .unwrap();

        assert_eq!(first_turn.arrival_time_ms, 0.0);
        assert_eq!(session_b.arrival_time_ms, 5.0);
        assert!(
            second_turn.arrival_time_ms >= first_turn.last_token_ms.unwrap() + 10.0,
            "follow-up turn should unlock after completion plus delay"
        );
    }

    #[test]
    fn test_concurrency_workload_delayed_follow_up_does_not_bypass_other_ready_sessions() {
        let (collector, _) = run_concurrency_workload_collect(
            &disagg_config(),
            multiturn_trace(),
            None,
            1,
            ReplayRouterMode::RoundRobin,
        );
        let mut input_lengths = collector
            .snapshots()
            .into_iter()
            .map(|snapshot| (snapshot.arrival_time_ms, snapshot.input_length))
            .collect::<Vec<_>>();
        input_lengths.sort_by(|left, right| left.0.total_cmp(&right.0));

        assert_eq!(
            input_lengths
                .into_iter()
                .map(|(_, input_length)| input_length)
                .collect::<Vec<_>>(),
            vec![64, 128, 192]
        );
    }
}
