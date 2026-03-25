// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use dynamo_kv_router::config::KvRouterConfig;
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::loadgen::{Trace, WorkloadDriver};
use crate::replay::router::ReplayRouter;
use crate::replay::{ReplayRouterMode, TraceSimulationReport, normalize_trace_requests};
use crate::scheduler::{AdmissionEvent, EngineScheduler, SchedulerHandle};

use super::demux::run_demux;
use super::state::{
    LiveReplayMode, LiveRuntimeStats, SharedLiveRuntimeStats, WorkloadDispatchState, now_ms,
    record_arrival,
};
use super::task::{RequestTaskContext, run_request_task, wait_for_workload_progress};

struct LiveRuntime {
    pending: VecDeque<DirectRequest>,
    senders: Arc<[mpsc::UnboundedSender<DirectRequest>]>,
    schedulers: Vec<EngineScheduler>,
    output_rx: mpsc::UnboundedReceiver<OutputSignal>,
    admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    cancel_token: CancellationToken,
    start: Instant,
    mode: LiveReplayMode,
    router: Arc<ReplayRouter>,
}

impl LiveRuntime {
    fn new(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        pending: VecDeque<DirectRequest>,
        num_workers: usize,
        mode: LiveReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        let cancel_token = CancellationToken::new();
        let (output_tx, output_rx) = mpsc::unbounded_channel();
        let (admission_tx, admission_rx) = mpsc::unbounded_channel();
        let router = Arc::new(ReplayRouter::new(
            router_mode,
            &args,
            router_config,
            num_workers,
        ));
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
            senders: Arc::from(senders),
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
        let senders = Arc::clone(&self.senders);
        let output_rx = self.output_rx;
        let admission_rx = self.admission_rx;
        let demux_stats = Arc::clone(&stats);
        let demux_router = Arc::clone(&router);
        let demux_task = tokio::spawn(async move {
            run_demux(
                start,
                arrival_rx,
                admission_rx,
                output_rx,
                demux_requests,
                demux_router,
                demux_stats,
            )
            .await
        });
        let mut tasks = JoinSet::new();
        let task_ctx = RequestTaskContext {
            senders,
            router: Arc::clone(&self.router),
            requests: Arc::clone(&requests),
            stats: Arc::clone(&stats),
            workload: None,
        };

        match self.mode {
            LiveReplayMode::Trace => {
                while let Some(request) = self.pending.pop_front() {
                    let arrival_ms = request.arrival_timestamp_ms.unwrap_or(0.0);
                    let deadline =
                        start + tokio::time::Duration::from_secs_f64(arrival_ms / 1000.0);
                    tokio::time::sleep_until(deadline).await;
                    record_arrival(&arrival_tx, &request, arrival_ms)?;
                    tasks.spawn(run_request_task(task_ctx.clone(), request, None));
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
                    record_arrival(&arrival_tx, &request, now_ms(start))?;
                    tasks.spawn(run_request_task(task_ctx.clone(), request, Some(permit)));
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

    async fn run_workload(
        mut self,
        driver: WorkloadDriver,
        total_turns: usize,
    ) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        let requests = Arc::new(DashMap::with_capacity(total_turns.max(1)));
        let stats = Arc::new(SharedLiveRuntimeStats::default());
        let (arrival_tx, arrival_rx) = mpsc::unbounded_channel();
        let demux_requests = Arc::clone(&requests);
        let start = self.start;
        let router = Arc::clone(&self.router);
        let senders = Arc::clone(&self.senders);
        let output_rx = self.output_rx;
        let admission_rx = self.admission_rx;
        let demux_stats = Arc::clone(&stats);
        let demux_router = Arc::clone(&router);
        let demux_task = tokio::spawn(async move {
            run_demux(
                start,
                arrival_rx,
                admission_rx,
                output_rx,
                demux_requests,
                demux_router,
                demux_stats,
            )
            .await
        });
        let workload = Arc::new(WorkloadDispatchState {
            driver: std::sync::Mutex::new(driver),
            wakeup: Notify::new(),
            start,
        });
        let mut tasks = JoinSet::new();
        let task_ctx = RequestTaskContext {
            senders,
            router: Arc::clone(&self.router),
            requests: Arc::clone(&requests),
            stats: Arc::clone(&stats),
            workload: Some(Arc::clone(&workload)),
        };
        let semaphore = match self.mode {
            LiveReplayMode::Trace => None,
            LiveReplayMode::Concurrency { max_in_flight } => {
                Some(Arc::new(Semaphore::new(max_in_flight)))
            }
        };

        loop {
            let now = now_ms(start);
            let dispatch_limit = match &semaphore {
                Some(semaphore) => semaphore.available_permits(),
                None => usize::MAX,
            };

            if dispatch_limit > 0 {
                let ready_turns = workload
                    .driver
                    .lock()
                    .unwrap()
                    .pop_ready(now, dispatch_limit);
                if !ready_turns.is_empty() {
                    for ready_turn in ready_turns {
                        let permit = match &semaphore {
                            Some(semaphore) => {
                                Some(semaphore.clone().try_acquire_owned().map_err(|_| {
                                    anyhow!(
                                        "online replay concurrency semaphore unexpectedly closed"
                                    )
                                })?)
                            }
                            None => None,
                        };
                        let arrival_at_ms = match self.mode {
                            LiveReplayMode::Trace => ready_turn.scheduled_ready_at_ms,
                            LiveReplayMode::Concurrency { .. } => now_ms(start),
                        };
                        record_arrival(&arrival_tx, &ready_turn.request, arrival_at_ms)?;
                        tasks.spawn(run_request_task(
                            task_ctx.clone(),
                            ready_turn.request,
                            permit,
                        ));
                    }
                    continue;
                }
            }

            let wake = workload.wakeup.notified();
            tokio::pin!(wake);
            let (is_drained, next_ready_ms) = {
                let mut driver = workload.driver.lock().unwrap();
                (driver.is_drained(), driver.next_ready_time_ms())
            };
            if is_drained {
                break;
            }

            wait_for_workload_progress(
                self.mode,
                semaphore.as_deref(),
                next_ready_ms,
                start,
                wake.as_mut(),
            )
            .await;
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
    router_config: Option<KvRouterConfig>,
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
        LiveRuntime::new(args, router_config, pending, num_workers, mode, router_mode)?
            .run()
            .await
    })
}

fn run_live_workload_runtime(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    driver: WorkloadDriver,
    total_turns: usize,
    num_workers: usize,
    mode: LiveReplayMode,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("failed to create online replay runtime: {e}"))?;

    runtime.block_on(async move {
        LiveRuntime::new(
            args,
            router_config,
            VecDeque::new(),
            num_workers,
            mode,
            router_mode,
        )?
        .run_workload(driver, total_turns)
        .await
    })
}

pub(crate) fn simulate_trace_requests(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (report, _) = run_live_runtime(
        args,
        router_config,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_requests(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
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
        router_config,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    let (report, _) = run_live_workload_runtime(
        args,
        router_config,
        trace.into_trace_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )?;
    Ok(report)
}

pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    let (report, _) = run_live_workload_runtime(
        args,
        router_config,
        trace.into_concurrency_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?;
    Ok(report)
}

#[cfg(test)]
pub(super) fn simulate_trace_requests_with_stats(
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
        None,
        pending,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )
}

#[cfg(test)]
pub(super) fn simulate_concurrency_requests_with_stats(
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
        None,
        pending,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
}

#[cfg(test)]
pub(super) fn simulate_trace_workload_with_stats(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    run_live_workload_runtime(
        args,
        None,
        trace.into_trace_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Trace,
        router_mode,
    )
}

#[cfg(test)]
pub(super) fn simulate_concurrency_workload_with_stats(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
    let args = args.normalized()?;
    let total_turns = trace
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    run_live_workload_runtime(
        args,
        None,
        trace.into_concurrency_driver()?,
        total_turns,
        num_workers,
        LiveReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
}
