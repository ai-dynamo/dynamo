// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::replay::router::ReplayRouter;
use crate::replay::{
    ReplayRouterMode, TraceCollector, TraceSimulationReport, normalize_trace_requests,
};
use crate::scheduler::{AdmissionEvent, EngineScheduler, SchedulerHandle};

#[derive(Clone, Copy, Debug)]
enum LiveReplayMode {
    Trace,
    Concurrency { max_in_flight: usize },
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct LiveRuntimeStats {
    pub(super) dispatch_history: Vec<usize>,
    pub(super) max_in_flight_seen: usize,
    pub(super) prefill_marked_count: usize,
    pub(super) freed_count: usize,
}

#[derive(Default)]
struct SharedLiveRuntimeStats {
    dispatch_history: Mutex<Vec<usize>>,
    current_in_flight: AtomicUsize,
    max_in_flight_seen: AtomicUsize,
    prefill_marked_count: AtomicUsize,
    freed_count: AtomicUsize,
}

impl SharedLiveRuntimeStats {
    fn record_dispatch(&self, worker_idx: usize) {
        self.dispatch_history.lock().unwrap().push(worker_idx);
        let current = self.current_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.max_in_flight_seen.fetch_max(current, Ordering::AcqRel);
    }

    fn record_completion(&self) {
        self.current_in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    fn record_prefill_marked(&self) {
        self.prefill_marked_count.fetch_add(1, Ordering::AcqRel);
    }

    fn record_freed(&self) {
        self.freed_count.fetch_add(1, Ordering::AcqRel);
    }

    fn snapshot(&self) -> LiveRuntimeStats {
        LiveRuntimeStats {
            dispatch_history: self.dispatch_history.lock().unwrap().clone(),
            max_in_flight_seen: self.max_in_flight_seen.load(Ordering::Acquire),
            prefill_marked_count: self.prefill_marked_count.load(Ordering::Acquire),
            freed_count: self.freed_count.load(Ordering::Acquire),
        }
    }
}

#[derive(Default)]
struct RequestSignals {
    first_token_seen: AtomicBool,
    completed_seen: AtomicBool,
    first_token_notify: Notify,
    completion_notify: Notify,
}

impl RequestSignals {
    fn notify_first_token(&self) {
        if self.first_token_seen.swap(true, Ordering::AcqRel) {
            return;
        }
        self.first_token_notify.notify_waiters();
    }

    fn notify_completion(&self) {
        if self.completed_seen.swap(true, Ordering::AcqRel) {
            return;
        }
        self.completion_notify.notify_waiters();
    }

    async fn wait_for_first_token(&self) {
        loop {
            let notified = self.first_token_notify.notified();
            if self.first_token_seen.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    async fn wait_for_completion(&self) {
        loop {
            let notified = self.completion_notify.notified();
            if self.completed_seen.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Copy)]
struct ArrivalEvent {
    uuid: Uuid,
    at_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
}

type RequestRegistry = Arc<DashMap<Uuid, Arc<RequestSignals>>>;

async fn run_demux(
    start: Instant,
    mut arrival_rx: mpsc::UnboundedReceiver<ArrivalEvent>,
    mut admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    mut output_rx: mpsc::UnboundedReceiver<OutputSignal>,
    requests: RequestRegistry,
) -> TraceSimulationReport {
    let mut collector = TraceCollector::default();
    let mut arrivals_open = true;
    let mut admissions_open = true;
    let mut outputs_open = true;

    loop {
        if !arrivals_open && !admissions_open && !outputs_open {
            break;
        }

        tokio::select! {
            biased;
            arrival = arrival_rx.recv(), if arrivals_open => {
                match arrival {
                    Some(arrival) => collector.on_arrival(
                        arrival.uuid,
                        arrival.at_ms,
                        arrival.input_tokens,
                        arrival.output_tokens,
                    ),
                    None => arrivals_open = false,
                }
            }
            admission = admission_rx.recv(), if admissions_open => {
                match admission {
                    Some(admission) => {
                        let now_ms = start.elapsed().as_secs_f64() * 1000.0;
                        collector.on_admit(admission.uuid, now_ms, admission.reused_input_tokens);
                    }
                    None => admissions_open = false,
                }
            }
            output = output_rx.recv(), if outputs_open => {
                match output {
                    Some(output) => {
                        let now_ms = start.elapsed().as_secs_f64() * 1000.0;
                        collector.on_token(output.uuid, now_ms);
                        if let Some(signals) = requests.get(&output.uuid) {
                            signals.notify_first_token();
                            if output.completed {
                                signals.notify_completion();
                            }
                        }
                    }
                    None => outputs_open = false,
                }
            }
        }
    }

    let wall_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    collector.finish().with_wall_time_ms(wall_time_ms)
}

struct LiveRuntime {
    pending: VecDeque<DirectRequest>,
    senders: Vec<mpsc::UnboundedSender<DirectRequest>>,
    schedulers: Vec<EngineScheduler>,
    output_rx: mpsc::UnboundedReceiver<OutputSignal>,
    admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    cancel_token: CancellationToken,
    start: Instant,
    mode: LiveReplayMode,
    router: Arc<ReplayRouter>,
}

fn now_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

struct DispatchContext<'a> {
    senders: &'a [mpsc::UnboundedSender<DirectRequest>],
    start: Instant,
    router: &'a Arc<ReplayRouter>,
    arrival_tx: &'a mpsc::UnboundedSender<ArrivalEvent>,
    requests: &'a RequestRegistry,
    stats: &'a Arc<SharedLiveRuntimeStats>,
    tasks: &'a mut JoinSet<Result<()>>,
}

async fn dispatch_request(
    ctx: DispatchContext<'_>,
    request: DirectRequest,
    permit: Option<OwnedSemaphorePermit>,
) -> Result<()> {
    let uuid = request
        .uuid
        .ok_or_else(|| anyhow!("online replay requires requests to have stable UUIDs"))?;
    let input_tokens = request.tokens.len();
    let output_tokens = request.max_output_tokens;
    let worker_idx = ctx
        .router
        .select_worker(&request, ctx.senders.len())
        .await?;
    if worker_idx >= ctx.senders.len() {
        bail!("online replay selected unknown worker index {worker_idx}");
    }

    let signals = Arc::new(RequestSignals::default());
    ctx.requests.insert(uuid, Arc::clone(&signals));
    if let Err(error) = ctx.senders[worker_idx].send(request) {
        ctx.requests.remove(&uuid);
        return Err(anyhow!(
            "online replay failed to dispatch request to worker {worker_idx}: {error}"
        ));
    }

    ctx.arrival_tx
        .send(ArrivalEvent {
            uuid,
            at_ms: now_ms(ctx.start),
            input_tokens,
            output_tokens,
        })
        .map_err(|_| anyhow!("online replay arrival channel closed"))?;

    ctx.stats.record_dispatch(worker_idx);
    let router = Arc::clone(ctx.router);
    let requests = Arc::clone(ctx.requests);
    let stats = Arc::clone(ctx.stats);
    ctx.tasks.spawn(async move {
        let result = async {
            signals.wait_for_first_token().await;
            if router.on_first_token(uuid).await? {
                stats.record_prefill_marked();
            }

            signals.wait_for_completion().await;
            if router.on_complete(uuid).await? {
                stats.record_freed();
            }
            Ok(())
        }
        .await;

        stats.record_completion();
        requests.remove(&uuid);
        drop(permit);
        result
    });

    Ok(())
}

impl LiveRuntime {
    fn new(
        args: MockEngineArgs,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: LiveReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        if pending.is_empty() {
            bail!("online replay requires at least one request");
        }

        let cancel_token = CancellationToken::new();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (admission_tx, admission_rx) = mpsc::unbounded_channel();
        let router = Arc::new(ReplayRouter::new(router_mode, &args, num_workers));
        let mut schedulers = Vec::with_capacity(num_workers);
        let mut senders = Vec::with_capacity(num_workers);

        for worker_idx in 0..num_workers {
            let scheduler = EngineScheduler::new_with_admission(
                args.clone(),
                0,
                Some(output_tx.clone()),
                router.sink(worker_idx as _),
                Some(cancel_token.clone()),
                Some(admission_tx.clone()),
            );
            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);
        }

        drop(output_tx);
        drop(admission_tx);

        Ok(Self {
            pending,
            senders,
            schedulers,
            output_rx,
            admission_rx,
            cancel_token,
            start: Instant::now(),
            mode,
            router,
        })
    }

    async fn run(mut self) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        let requests = Arc::new(DashMap::with_capacity(self.pending.len()));
        let stats = Arc::new(SharedLiveRuntimeStats::default());
        let (arrival_tx, arrival_rx) = mpsc::unbounded_channel();
        let demux_requests = Arc::clone(&requests);
        let start = self.start;
        let router = Arc::clone(&self.router);
        let senders = self.senders.clone();
        let output_rx = self.output_rx;
        let admission_rx = self.admission_rx;
        let demux_task = tokio::spawn(async move {
            run_demux(start, arrival_rx, admission_rx, output_rx, demux_requests).await
        });
        let mut tasks = JoinSet::new();

        match self.mode {
            LiveReplayMode::Trace => {
                while let Some(request) = self.pending.pop_front() {
                    let arrival_ms = request.arrival_timestamp_ms.unwrap_or(0.0);
                    let deadline =
                        start + tokio::time::Duration::from_secs_f64(arrival_ms / 1000.0);
                    tokio::time::sleep_until(deadline).await;
                    dispatch_request(
                        DispatchContext {
                            senders: &senders,
                            start,
                            router: &router,
                            arrival_tx: &arrival_tx,
                            requests: &requests,
                            stats: &stats,
                            tasks: &mut tasks,
                        },
                        request,
                        None,
                    )
                    .await?;
                }
            }
            LiveReplayMode::Concurrency { max_in_flight } => {
                let semaphore = Arc::new(Semaphore::new(max_in_flight));
                while let Some(request) = self.pending.pop_front() {
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|_| anyhow!("online replay concurrency semaphore closed"))?;
                    dispatch_request(
                        DispatchContext {
                            senders: &senders,
                            start,
                            router: &router,
                            arrival_tx: &arrival_tx,
                            requests: &requests,
                            stats: &stats,
                            tasks: &mut tasks,
                        },
                        request,
                        Some(permit),
                    )
                    .await?;
                }
            }
        }

        while let Some(result) = tasks.join_next().await {
            result.map_err(|e| anyhow!("online replay request task failed: {e}"))??;
        }

        drop(arrival_tx);
        self.cancel_token.cancel();
        self.schedulers.clear();

        let report = demux_task
            .await
            .map_err(|e| anyhow!("online replay demux task failed: {e}"))?;
        router.shutdown().await?;
        Ok((report, stats.snapshot()))
    }
}

fn run_live_runtime(
    args: MockEngineArgs,
    pending: VecDeque<DirectRequest>,
    num_workers: usize,
    mode: LiveReplayMode,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(args, pending, num_workers, mode, router_mode)?
            .run()
            .await
    })
}

pub(crate) fn simulate_trace_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (report, _) = run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_requests(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    if requests.is_empty() {
        bail!("online concurrency replay requires at least one request");
    }

    let pending = VecDeque::from(requests);
    let (report, _) = run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?;
    Ok(report)
}

#[cfg(test)]
fn simulate_trace_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )
}

#[cfg(test)]
fn simulate_concurrency_requests_with_stats(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    run_live_runtime(
        args,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::protocols::{DirectRequest, EngineType, SglangArgs};

    fn replay_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .speedup_ratio(1000.0)
            .block_size(64)
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

    fn request(uuid: u128, token: u32, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
        DirectRequest {
            tokens: vec![token; 64],
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms,
        }
    }

    #[test]
    fn test_online_trace_replay_single_worker_completes() {
        let args = replay_args();
        let requests = vec![request(1, 11, Some(0.0)), request(2, 22, Some(1.0))];

        let report =
            simulate_trace_requests(args, requests, 1, 1.0, ReplayRouterMode::RoundRobin).unwrap();

        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 4);
        assert!(report.throughput.wall_time_ms >= 0.0);
    }

    #[test]
    fn test_online_trace_replay_uses_round_robin_dispatch() {
        let args = replay_args();
        let requests = vec![
            request(1, 1, Some(0.0)),
            request(2, 2, Some(0.0)),
            request(3, 3, Some(0.0)),
            request(4, 4, Some(0.0)),
            request(5, 5, Some(0.0)),
        ];

        let (_, stats) = simulate_trace_requests_with_stats(
            args,
            requests,
            3,
            1.0,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        assert_eq!(stats.dispatch_history, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_online_concurrency_replay_respects_max_in_flight() {
        let args = replay_args();
        let requests = vec![
            request(1, 10, None),
            request(2, 20, None),
            request(3, 30, None),
            request(4, 40, None),
        ];

        let (report, stats) = simulate_concurrency_requests_with_stats(
            args,
            requests,
            2,
            2,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap();

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(stats.max_in_flight_seen, 2);
    }

    #[test]
    fn test_online_trace_replay_populates_admit_reuse_stats() {
        let args = replay_args();
        let requests = vec![request(1, 77, Some(0.0)), request(2, 77, Some(5.0))];

        let report =
            simulate_trace_requests(args, requests, 1, 1.0, ReplayRouterMode::RoundRobin).unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        assert!(report.prefix_cache_reused_ratio > 0.0);
    }

    #[test]
    fn test_online_trace_replay_kv_router_prefers_cached_worker() {
        let args = replay_args();
        let requests = vec![request(1, 88, Some(0.0)), request(2, 88, Some(500.0))];

        let (_, stats) =
            simulate_trace_requests_with_stats(args, requests, 2, 1.0, ReplayRouterMode::KvRouter)
                .unwrap();

        assert_eq!(stats.dispatch_history.len(), 2);
        assert_eq!(stats.dispatch_history[0], stats.dispatch_history[1]);
    }

    #[test]
    fn test_online_trace_replay_sglang_single_worker_completes() {
        let args = sglang_replay_args();
        let requests = vec![request(101, 7, Some(0.0)), request(102, 8, Some(1.0))];

        let report =
            simulate_trace_requests(args, requests, 1, 1.0, ReplayRouterMode::RoundRobin).unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.request_counts.total_output_tokens, 4);
    }

    #[test]
    fn test_online_trace_replay_sglang_kv_router_smoke() {
        let args = sglang_replay_args();
        let requests = vec![request(111, 9, Some(0.0)), request(112, 9, Some(500.0))];

        let (report, stats) =
            simulate_trace_requests_with_stats(args, requests, 2, 1.0, ReplayRouterMode::KvRouter)
                .unwrap();

        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(stats.dispatch_history.len(), 2);
    }

    #[test]
    fn test_online_concurrency_replay_kv_router_respects_max_in_flight() {
        let args = replay_args();
        let requests = vec![
            request(1, 10, None),
            request(2, 20, None),
            request(3, 10, None),
            request(4, 20, None),
        ];

        let (report, stats) = simulate_concurrency_requests_with_stats(
            args,
            requests,
            2,
            2,
            ReplayRouterMode::KvRouter,
        )
        .unwrap();

        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(stats.max_in_flight_seen, 2);
    }

    #[test]
    fn test_online_trace_replay_kv_router_marks_prefill_and_free_once() {
        let args = replay_args();
        let requests = vec![DirectRequest {
            tokens: vec![9; 64],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(9)),
            dp_rank: 0,
            arrival_timestamp_ms: Some(0.0),
        }];

        let (_, stats) =
            simulate_trace_requests_with_stats(args, requests, 1, 1.0, ReplayRouterMode::KvRouter)
                .unwrap();

        assert_eq!(stats.prefill_marked_count, 1);
        assert_eq!(stats.freed_count, 1);
    }
}
