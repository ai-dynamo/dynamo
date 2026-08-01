// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Result, bail};
use tokio::sync::{Notify, Semaphore, mpsc, watch};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{DirectRequest, FpmPublisher};
use crate::live::{LiveEngine, LiveEngineOptions, ObservedAdmission};
use crate::loadgen::WorkloadDriver;
use crate::replay::TraceSimulationReport;

use super::ReplayRouter;
use super::entrypoints::OnlineReplayConfig;
use super::recorder::{OnlineRecorderOptions, OnlineTraceRecorder, forward_admissions};
use super::state::{
    LiveReplayMode, LiveRuntimeStats, SharedLiveRuntimeStats, WorkloadDispatchState, arrival_event,
    now_ms,
};
use super::task::{
    InFlightGuard, RequestTaskContext, run_request_task, wait_for_workload_progress,
};

pub(super) struct LiveRuntime {
    pending: std::collections::VecDeque<DirectRequest>,
    // Worker-major rank handles: worker_idx * dp_size + dp_rank.
    engines: Arc<[LiveEngine]>,
    num_workers: usize,
    dp_size: usize,
    admission_rx: mpsc::UnboundedReceiver<ObservedAdmission>,
    start: Instant,
    mode: LiveReplayMode,
    router: Arc<ReplayRouter>,
    recorder_options: OnlineRecorderOptions,
    cancel: CancellationToken,
}

struct LiveRunSession {
    task_ctx: RequestTaskContext,
    tasks: JoinSet<Result<()>>,
    recorder_tx: super::recorder::RecorderSender,
    recorder: OnlineTraceRecorder,
    admission_task: tokio::task::JoinHandle<Result<()>>,
}

struct LiveRunSessionConfig {
    engines: Arc<[LiveEngine]>,
    num_workers: usize,
    dp_size: usize,
    router: Arc<ReplayRouter>,
    start: Instant,
    workload: Option<Arc<WorkloadDispatchState>>,
    cancel: CancellationToken,
}

impl LiveRunSession {
    fn new(
        config: LiveRunSessionConfig,
        admission_rx: mpsc::UnboundedReceiver<ObservedAdmission>,
        recorder_options: OnlineRecorderOptions,
    ) -> Self {
        let LiveRunSessionConfig {
            engines,
            num_workers,
            dp_size,
            router,
            start,
            workload,
            cancel,
        } = config;
        let recorder = OnlineTraceRecorder::start(recorder_options);
        let recorder_tx = recorder.sender();
        let admission_task =
            tokio::spawn(forward_admissions(start, admission_rx, recorder.sender()));
        let task_ctx = RequestTaskContext {
            engines,
            num_workers,
            dp_size,
            router,
            recorder: recorder_tx.clone(),
            stats: Arc::new(SharedLiveRuntimeStats::default()),
            workload,
            cancel,
            start,
        };
        Self {
            task_ctx,
            tasks: JoinSet::new(),
            recorder_tx,
            recorder,
            admission_task,
        }
    }

    async fn finish(mut self) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        while !self.tasks.is_empty() {
            tokio::select! {
                biased;
                _ = self.task_ctx.cancel.cancelled() => bail!("online replay cancelled"),
                joined = self.tasks.join_next() => {
                    if let Some(joined) = joined {
                        joined??;
                    }
                }
            }
        }
        if self.task_ctx.cancel.is_cancelled() {
            bail!("online replay cancelled");
        }

        let LiveRunSession {
            task_ctx,
            recorder_tx,
            recorder,
            admission_task,
            ..
        } = self;
        let wall_time_ms = now_ms(task_ctx.start);
        let vllm_preemptions_total = task_ctx
            .engines
            .iter()
            .map(|engine| engine.metrics_receiver().borrow().vllm_preemptions_total)
            .sum();
        let stats_snapshot = task_ctx.stats.snapshot(vllm_preemptions_total);
        let engines = Arc::clone(&task_ctx.engines);
        let router = Arc::clone(&task_ctx.router);
        let cancel = task_ctx.cancel.clone();
        drop(task_ctx);
        drop(recorder_tx);

        for engine in engines.iter() {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => bail!("online replay cancelled"),
                result = engine.shutdown() => result?,
            }
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => bail!("online replay cancelled"),
            result = admission_task => result??,
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => bail!("online replay cancelled"),
            result = router.shutdown() => result?,
        }
        let report = tokio::select! {
            biased;
            _ = cancel.cancelled() => bail!("online replay cancelled"),
            result = recorder.finish(wall_time_ms) => result?,
        };
        if cancel.is_cancelled() {
            bail!("online replay cancelled");
        }
        Ok((report, stats_snapshot))
    }
}

impl LiveRuntime {
    /// Build the shared router and one grouped live engine per logical replay worker.
    pub(super) fn new(
        config: OnlineReplayConfig,
        pending: std::collections::VecDeque<DirectRequest>,
        mode: LiveReplayMode,
        cancel: CancellationToken,
    ) -> Result<Self> {
        Self::new_inner(config, pending, mode, None, cancel)
    }

    #[cfg(test)]
    pub(super) fn new_with_output_gate(
        config: OnlineReplayConfig,
        pending: std::collections::VecDeque<DirectRequest>,
        mode: LiveReplayMode,
        output_gate: watch::Receiver<bool>,
        cancel: CancellationToken,
    ) -> Result<Self> {
        Self::new_inner(config, pending, mode, Some(output_gate), cancel)
    }

    fn new_inner(
        config: OnlineReplayConfig,
        pending: std::collections::VecDeque<DirectRequest>,
        mode: LiveReplayMode,
        output_gate: Option<watch::Receiver<bool>>,
        cancel: CancellationToken,
    ) -> Result<Self> {
        let OnlineReplayConfig {
            args,
            router_config,
            prefill_load_estimator,
            num_workers,
            router_mode,
            options: replay_options,
        } = config;
        let recorder_options = OnlineRecorderOptions {
            capture_per_request: replay_options.record_per_request,
            sla: replay_options.sla,
            num_workers,
            gpus_per_worker: args.aic_gpus_per_worker(),
        };
        let (admission_tx, admission_rx) = mpsc::unbounded_channel();
        let dp_size = usize::try_from(args.dp_size)
            .map_err(|_| anyhow::anyhow!("attention-DP size does not fit into usize"))?;
        anyhow::ensure!(
            num_workers > 0,
            "online replay requires at least one worker"
        );
        anyhow::ensure!(dp_size > 0, "online replay requires at least one DP rank");
        let router = Arc::new(ReplayRouter::new(
            router_mode,
            &args,
            router_config,
            prefill_load_estimator,
            num_workers,
        )?);
        let rank_handle_count = num_workers
            .checked_mul(dp_size)
            .ok_or_else(|| anyhow::anyhow!("online replay rank-handle count overflow"))?;
        let mut engines = Vec::with_capacity(rank_handle_count);
        for worker_idx in 0..num_workers {
            let rank_options = (0..dp_size)
                .map(|_| LiveEngineOptions {
                    kv_event_publishers: router.sink(worker_idx as _),
                    admission_tx: Some(admission_tx.clone()),
                    fpm_publisher: FpmPublisher::default(),
                    request_output_capacity: None,
                    allow_zero_output: true,
                })
                .collect();
            engines.extend(LiveEngine::start_grouped_with_options(
                args.clone(),
                rank_options,
                output_gate.clone(),
            )?);
        }
        drop(admission_tx);

        Ok(Self {
            pending,
            engines: Arc::from(engines),
            num_workers,
            dp_size,
            admission_rx,
            start: Instant::now(),
            mode,
            router,
            recorder_options,
            cancel,
        })
    }

    #[cfg(test)]
    pub(super) fn engines(&self) -> Arc<[LiveEngine]> {
        Arc::clone(&self.engines)
    }

    #[cfg(test)]
    pub(super) fn router(&self) -> Arc<ReplayRouter> {
        Arc::clone(&self.router)
    }

    /// Replay a finite queue of requests and return the final trace report plus debug stats.
    pub(super) async fn run(self) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        let LiveRuntime {
            mut pending,
            engines,
            num_workers,
            dp_size,
            admission_rx,
            start,
            mode,
            router,
            recorder_options,
            cancel,
        } = self;
        let session_config = LiveRunSessionConfig {
            engines,
            num_workers,
            dp_size,
            router,
            workload: None,
            cancel,
            start,
        };
        let mut session = LiveRunSession::new(session_config, admission_rx, recorder_options);

        match mode {
            LiveReplayMode::Trace => {
                while let Some(request) = pending.pop_front() {
                    let arrival_ms = request.arrival_timestamp_ms.unwrap_or(0.0);
                    let deadline =
                        start + tokio::time::Duration::from_secs_f64(arrival_ms / 1000.0);
                    loop {
                        tokio::select! {
                            biased;
                            _ = session.task_ctx.cancel.cancelled() => {
                                bail!("online replay cancelled");
                            }
                            joined = session.tasks.join_next(), if !session.tasks.is_empty() => {
                                if let Some(joined) = joined {
                                    joined??;
                                }
                            }
                            _ = tokio::time::sleep_until(deadline) => break,
                        }
                    }
                    if session.task_ctx.cancel.is_cancelled() {
                        bail!("online replay cancelled");
                    }
                    let arrival = arrival_event(&request, arrival_ms)?;
                    session.recorder_tx.record_arrival(arrival)?;
                    if session.task_ctx.cancel.is_cancelled() {
                        bail!("online replay cancelled");
                    }
                    session
                        .tasks
                        .spawn(run_request_task(session.task_ctx.clone(), request, None));
                }
            }
            LiveReplayMode::Concurrency { max_in_flight } => {
                let semaphore = Arc::new(Semaphore::new(max_in_flight));
                while let Some(request) = pending.pop_front() {
                    let acquire = semaphore.clone().acquire_owned();
                    tokio::pin!(acquire);
                    let permit = loop {
                        tokio::select! {
                            biased;
                            _ = session.task_ctx.cancel.cancelled() => {
                                bail!("online replay cancelled");
                            }
                            joined = session.tasks.join_next(), if !session.tasks.is_empty() => {
                                if let Some(joined) = joined {
                                    joined??;
                                }
                            }
                            permit = &mut acquire => {
                                break permit?;
                            }
                        }
                    };
                    let arrival = arrival_event(&request, now_ms(start))?;
                    session.recorder_tx.record_arrival(arrival)?;
                    if session.task_ctx.cancel.is_cancelled() {
                        bail!("online replay cancelled");
                    }
                    let task_ctx = session.task_ctx.clone();
                    session.tasks.spawn(async move {
                        let _permit = permit;
                        run_request_task(task_ctx, request, None).await
                    });
                }
            }
        }

        session.finish().await
    }

    /// Drive a multi-turn workload driver until it is drained and all spawned request tasks finish.
    pub(super) async fn run_workload(
        self,
        driver: WorkloadDriver,
        _total_turns: usize,
    ) -> Result<(TraceSimulationReport, LiveRuntimeStats)> {
        let LiveRuntime {
            engines,
            num_workers,
            dp_size,
            admission_rx,
            start,
            mode,
            router,
            recorder_options,
            cancel,
            ..
        } = self;
        let cap_enabled = matches!(mode, LiveReplayMode::Concurrency { .. });
        let workload = Arc::new(WorkloadDispatchState {
            driver: std::sync::Mutex::new(driver),
            wakeup: Notify::new(),
            start,
        });
        let session_config = LiveRunSessionConfig {
            engines,
            num_workers,
            dp_size,
            router,
            workload: Some(Arc::clone(&workload)),
            cancel,
            start,
        };
        let mut session = LiveRunSession::new(session_config, admission_rx, recorder_options);

        loop {
            if session.task_ctx.cancel.is_cancelled() {
                bail!("online replay cancelled");
            }
            while let Some(joined) = session.tasks.try_join_next() {
                joined??;
            }

            let now = now_ms(start);
            let ready_turns =
                tokio::task::block_in_place(|| workload.driver.lock().unwrap().pop_ready(now, 1));
            if let Some(ready_turn) = ready_turns.into_iter().next() {
                let guard = cap_enabled
                    .then(|| InFlightGuard::new(Arc::clone(&workload), ready_turn.request_uuid));
                let arrival_at_ms = match mode {
                    LiveReplayMode::Trace => ready_turn.scheduled_ready_at_ms,
                    LiveReplayMode::Concurrency { .. } => now_ms(start),
                };
                let arrival = arrival_event(&ready_turn.request, arrival_at_ms)?;
                session.recorder_tx.record_arrival(arrival)?;
                if ready_turn.emit_session_metadata {
                    session.recorder_tx.record_session_metadata(
                        ready_turn.request_uuid,
                        ready_turn.session_id,
                        ready_turn.turn_index,
                    )?;
                }
                if session.task_ctx.cancel.is_cancelled() {
                    bail!("online replay cancelled");
                }
                session.tasks.spawn(run_request_task(
                    session.task_ctx.clone(),
                    ready_turn.request,
                    guard,
                ));
                tokio::task::yield_now().await;
                continue;
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

            tokio::select! {
                biased;
                _ = session.task_ctx.cancel.cancelled() => {
                    bail!("online replay cancelled");
                }
                joined = session.tasks.join_next(), if !session.tasks.is_empty() => {
                    if let Some(joined) = joined {
                        joined??;
                    }
                }
                _ = wait_for_workload_progress(next_ready_ms, start, wake.as_mut()) => {}
            }
        }

        session.finish().await
    }
}
