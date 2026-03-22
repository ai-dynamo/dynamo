// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::events::{SimulationEvent, SimulationEventKind};
use super::normalize_trace_requests;
use super::state::OfflineWorkerState;
use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::replay::router::OfflineReplayRouter;
use crate::replay::{ReplayRouterMode, TraceCollector, TraceSimulationReport};
use crate::scheduler::RouterEventVisibility;
use anyhow::bail;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::RouterEvent;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
enum ReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[cfg(test)]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct OfflineRuntimeStats {
    dispatch_history: Vec<usize>,
    dispatch_order: Vec<Uuid>,
    assigned_worker_by_uuid: HashMap<Uuid, usize>,
    max_in_flight_seen: usize,
    prefill_marked_count: usize,
    freed_count: usize,
    max_router_pending: usize,
}

#[cfg(not(test))]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct OfflineRuntimeStats;

struct OfflineRuntime {
    now_ms: f64,
    next_worker_idx: usize,
    next_event_seq: u64,
    pending: VecDeque<DirectRequest>,
    router_pending: HashMap<Uuid, DirectRequest>,
    workers: Vec<OfflineWorkerState>,
    collector: TraceCollector,
    events: BinaryHeap<SimulationEvent>,
    mode: ReplayMode,
    router: Option<OfflineReplayRouter>,
    prefill_completed: HashSet<Uuid>,
    stats: OfflineRuntimeStats,
}

impl OfflineRuntime {
    fn new(
        args: &MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        pending: VecDeque<DirectRequest>,
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
            pending,
            router_pending: HashMap::new(),
            workers: (0..num_workers)
                .map(|worker_idx| {
                    OfflineWorkerState::new(worker_idx, args.clone(), capture_kv_events)
                })
                .collect(),
            collector: TraceCollector::default(),
            events: BinaryHeap::new(),
            mode,
            router,
            prefill_completed: HashSet::new(),
            #[cfg(test)]
            stats: OfflineRuntimeStats::default(),
            #[cfg(not(test))]
            stats: OfflineRuntimeStats,
        })
    }

    fn cluster_in_flight(&self) -> usize {
        self.workers
            .iter()
            .map(OfflineWorkerState::in_flight)
            .sum::<usize>()
            + self.router_pending.len()
    }

    fn record_in_flight_peak(&mut self) {
        #[cfg(test)]
        {
            self.stats.max_in_flight_seen =
                self.stats.max_in_flight_seen.max(self.cluster_in_flight());
        }
    }

    fn record_router_pending(&mut self) {
        #[cfg(test)]
        let Some(router) = self.router.as_ref() else {
            return;
        };
        #[cfg(test)]
        {
            self.stats.max_router_pending =
                self.stats.max_router_pending.max(router.pending_count());
        }
    }

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

    fn validate_worker_idx(&self, worker_idx: usize) -> anyhow::Result<()> {
        if worker_idx >= self.workers.len() {
            bail!("offline replay selected unknown worker index {worker_idx}");
        }
        Ok(())
    }

    fn dispatch_to_worker(
        &mut self,
        request: DirectRequest,
        uuid: Uuid,
        worker_idx: usize,
    ) -> anyhow::Result<()> {
        self.validate_worker_idx(worker_idx)?;
        self.workers[worker_idx].receive_request(request);
        self.record_dispatch(uuid, worker_idx);
        Ok(())
    }

    fn dispatch_router_admissions(&mut self, admissions: Vec<(Uuid, usize)>) -> anyhow::Result<()> {
        for (uuid, worker_idx) in admissions {
            let request = self.router_pending.remove(&uuid).ok_or_else(|| {
                anyhow::anyhow!("offline replay missing queued request state for {uuid}")
            })?;
            self.dispatch_to_worker(request, uuid, worker_idx)?;
        }
        Ok(())
    }

    fn assign_request(
        &mut self,
        mut request: DirectRequest,
        arrival_time_ms: f64,
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

        let Some(router) = self.router.as_mut() else {
            let worker_idx = self.next_worker_idx;
            self.next_worker_idx = (self.next_worker_idx + 1) % self.workers.len();
            self.dispatch_to_worker(request, uuid, worker_idx)?;
            return Ok(uuid);
        };

        let maybe_worker_idx = router.submit_request(&request, self.now_ms)?;
        self.record_router_pending();
        if let Some(worker_idx) = maybe_worker_idx {
            self.dispatch_to_worker(request, uuid, worker_idx)?;
            return Ok(uuid);
        }

        self.router_pending.insert(uuid, request);
        self.record_in_flight_peak();
        Ok(uuid)
    }

    fn is_done(&self) -> bool {
        self.pending.is_empty()
            && self.events.is_empty()
            && self.cluster_in_flight() == 0
            && self.workers.iter().all(OfflineWorkerState::is_drained)
    }

    fn next_timestamp(&self) -> Option<f64> {
        let next_event_ms = self.events.peek().map(|event| event.at_ms);
        let next_arrival_ms = match self.mode {
            ReplayMode::Trace => self
                .pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            ReplayMode::Concurrency { .. } => None,
        };

        match (next_arrival_ms, next_event_ms) {
            (Some(arrival_ms), Some(event_ms)) => Some(arrival_ms.min(event_ms)),
            (Some(arrival_ms), None) => Some(arrival_ms),
            (None, Some(event_ms)) => Some(event_ms),
            (None, None) => None,
        }
    }

    fn push_event(&mut self, at_ms: f64, kind: SimulationEventKind) {
        self.events.push(SimulationEvent {
            at_ms,
            seq_no: self.next_event_seq,
            kind,
        });
        self.next_event_seq += 1;
    }

    fn apply_completed_requests(&mut self, worker_idx: usize, completed_requests: usize) {
        self.workers[worker_idx].mark_completed(completed_requests);
    }

    fn apply_router_events(&mut self, events: Vec<RouterEvent>) -> anyhow::Result<()> {
        let Some(router) = self.router.as_mut() else {
            return Ok(());
        };
        for event in events {
            router.apply_event(event)?;
        }
        Ok(())
    }

    fn process_output_signal(&mut self, signal: OutputSignal) -> anyhow::Result<()> {
        let mut admissions = Vec::new();
        if signal.completed {
            if let Some(router) = self.router.as_mut() {
                admissions = router.free(signal.uuid)?;
                #[cfg(test)]
                {
                    self.stats.freed_count += 1;
                }
                self.record_router_pending();
            }
            self.prefill_completed.remove(&signal.uuid);
            self.dispatch_router_admissions(admissions)?;
            return Ok(());
        }

        if !self.prefill_completed.insert(signal.uuid) {
            return Ok(());
        }

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

    fn apply_worker_completions(&mut self) -> anyhow::Result<bool> {
        let mut changed = false;
        loop {
            let Some(event) = self.events.peek() else {
                break;
            };
            if event.at_ms != self.now_ms {
                break;
            }
            if !matches!(event.kind, SimulationEventKind::WorkerCompletion { .. }) {
                break;
            }

            let event = self.events.pop().expect("event must exist after peek");
            let SimulationEventKind::WorkerCompletion {
                worker_idx,
                completed_requests,
                output_signals,
                kv_events,
            } = event.kind;

            self.workers[worker_idx].mark_idle();
            self.process_completed_pass(worker_idx, completed_requests, output_signals, kv_events)?;
            changed = true;
        }

        Ok(changed)
    }

    fn release_trace_arrivals(&mut self) -> anyhow::Result<bool> {
        let mut released_any = false;
        while self
            .pending
            .front()
            .and_then(|request| request.arrival_timestamp_ms)
            .is_some_and(|arrival_ms| arrival_ms <= self.now_ms)
        {
            let request = self
                .pending
                .pop_front()
                .expect("front request must exist when arrival is ready");
            let arrival_ms = request
                .arrival_timestamp_ms
                .expect("trace replay requests must have an arrival timestamp");
            self.assign_request(request, arrival_ms)?;
            released_any = true;
        }

        Ok(released_any)
    }

    fn top_off_concurrency(&mut self, max_in_flight: usize) -> anyhow::Result<bool> {
        let mut released_any = false;
        while self.cluster_in_flight() < max_in_flight {
            let Some(request) = self.pending.pop_front() else {
                break;
            };
            self.assign_request(request, self.now_ms)?;
            released_any = true;
        }

        Ok(released_any)
    }

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
                self.push_event(
                    executed.end_ms,
                    SimulationEventKind::WorkerCompletion {
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

    fn run(mut self) -> anyhow::Result<(TraceCollector, OfflineRuntimeStats)> {
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

        if let Some(router) = self.router.as_mut() {
            router.shutdown();
        }

        Ok((self.collector, self.stats))
    }
}

pub(crate) fn simulate_trace_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = OfflineRuntime::new(
        &args,
        router_config,
        pending,
        num_workers,
        ReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let (collector, _) = OfflineRuntime::new(
        &args,
        router_config,
        pending,
        num_workers,
        ReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

#[cfg(test)]
fn run_trace_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, OfflineRuntimeStats) {
    let pending = normalize_trace_requests(requests, 1.0).unwrap();
    OfflineRuntime::new(
        args,
        None,
        pending,
        num_workers,
        ReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
fn run_concurrency_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, OfflineRuntimeStats) {
    OfflineRuntime::new(
        args,
        None,
        VecDeque::from(requests),
        num_workers,
        ReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::super::single::{run_concurrency_single_collect, run_trace_single_collect};
    use super::*;
    use crate::common::protocols::{EngineType, SglangArgs};
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
        assert_eq!(stats.freed_count, 2);
        assert_eq!(stats.max_router_pending, 0);
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

        assert!(stats.max_router_pending > 0);
        assert!(request_3.first_admit_ms.unwrap() > request_3.arrival_time_ms);
        assert!(
            request_3.first_admit_ms.unwrap()
                < request_1
                    .last_token_ms
                    .unwrap()
                    .min(request_2.last_token_ms.unwrap())
        );
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

        assert!(fcfs_stats.max_router_pending > 0);
        assert!(lcfs_stats.max_router_pending > 0);
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
        assert!(stats.max_router_pending > 0);
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
        assert_eq!(stats.max_router_pending, 0);
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
        assert_eq!(stats.max_router_pending, 0);
        for uuid in [11_u128, 22, 33] {
            assert_eq!(
                multi.snapshot(Uuid::from_u128(uuid)),
                single.snapshot(Uuid::from_u128(uuid))
            );
        }
    }
}
