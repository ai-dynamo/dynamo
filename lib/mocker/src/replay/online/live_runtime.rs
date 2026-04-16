// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use dynamo_kv_router::config::KvRouterConfig;
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{DirectRequest, FpmPublisher, MockEngineArgs, OutputSignal};
use crate::loadgen::WorkloadDriver;
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport};
use crate::scheduler::{AdmissionEvent, EngineScheduler, SchedulerHandle};

use super::ReplayRouter;
use super::demux::run_demux;
use super::state::{
    LiveReplayMode, LiveRuntimeStats, SharedLiveRuntimeStats, WorkloadDispatchState, now_ms,
    record_arrival,
};
use super::task::{RequestTaskContext, run_request_task, wait_for_workload_progress};

pub(super) struct LiveRuntime {
    pending: std::collections::VecDeque<DirectRequest>,
    senders: Arc<[mpsc::UnboundedSender<DirectRequest>]>,
    schedulers: Vec<EngineScheduler>,
    output_rx: mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    cancel_token: CancellationToken,
    start: Instant,
    mode: LiveReplayMode,
    router: Arc<ReplayRouter>,
}

impl LiveRuntime {
    /// Build the shared router, worker schedulers, and demux inputs for one live replay run.
    pub(super) fn new(
        args: MockEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        pending: std::collections::VecDeque<DirectRequest>,
        num_workers: usize,
        mode: LiveReplayMode,
        router_mode: ReplayRouterMode,
    ) -> Result<Self> {
        let cancel_token = CancellationToken::new();
        let (output_tx, output_rx) = mpsc::unbounded_channel::<Vec<OutputSignal>>();
        let (admission_tx, admission_rx) = mpsc::unbounded_channel();
        let router = Arc::new(ReplayRouter::new(
            router_mode,
            &args,
            router_config,
            prefill_load_estimator,
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
                FpmPublisher::default(),
            );
            senders.push(scheduler.request_sender());
            schedulers.push(scheduler);
        }
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

    /// Replay a finite queue of requests and return the final trace report plus debug stats.
    pub(super) async fn run(mut self) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
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

    /// Drive a multi-turn workload driver until it is drained and all spawned request tasks finish.
    pub(super) async fn run_workload(
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
