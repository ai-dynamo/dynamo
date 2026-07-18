// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::common::utils::sleep_until_precise;
use crate::scheduler::{
    AdmissionEvent, LiveBoundaryCore, LiveEffectsPublisher, MockerMetrics, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerHandle, SchedulerLifecycleEvent,
    capture_deferred_kv_publish_sink, handoff_channel_capacity,
};

use super::core::SglangCore;

#[derive(Clone)]
pub struct SglangScheduler {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    lifecycle_rx: Arc<Mutex<Option<mpsc::Receiver<SchedulerLifecycleEvent>>>>,
    metrics_rx: tokio::sync::watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<CancelGuard>,
}

struct CancelGuard(CancellationToken);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl SglangScheduler {
    pub fn new(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            None,
            fpm_publisher,
        )
    }

    pub(crate) fn new_with_admission(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        Self::new_internal(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            admission_tx,
            fpm_publisher,
        )
    }

    fn new_internal(
        args: MockEngineArgs,
        dp_rank: u32,
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        kv_event_publishers: KvEventPublishers,
        cancellation_token: Option<CancellationToken>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        fpm_publisher: FpmPublisher,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<DirectRequest>();
        let control_capacity = handoff_channel_capacity(&args);
        let (command_tx, mut command_rx) =
            mpsc::channel::<SchedulerCommandEnvelope>(control_capacity);
        let (lifecycle_tx, lifecycle_rx) =
            mpsc::channel::<SchedulerLifecycleEvent>(control_capacity);
        let total_blocks = args.num_gpu_blocks as u64;
        let initial_metrics = MockerMetrics::new(dp_rank, 0, total_blocks);
        let (metrics_tx, metrics_rx) =
            tokio::sync::watch::channel::<MockerMetrics>(initial_metrics);

        let cancel_token = cancellation_token.unwrap_or_default();
        let cancel_token_clone = cancel_token.clone();
        let cancel_guard = Arc::new(CancelGuard(cancel_token));
        // Ordinary live users also need the ordered command channel for
        // request cancellation. Disaggregated workers additionally use it for
        // handoff lifecycle commands.
        let controls_enabled = true;

        tokio::spawn(async move {
            let (deferred_kv_events, buffering_publishers) = capture_deferred_kv_publish_sink(
                !kv_event_publishers.is_empty(),
                kv_event_publishers.raw_enabled(),
            );
            let mut core = SglangCore::new_with_sink(args, dp_rank, buffering_publishers);
            let publisher = LiveEffectsPublisher::new(
                output_tx,
                admission_tx,
                lifecycle_tx,
                metrics_tx,
                kv_event_publishers,
                fpm_publisher,
                deferred_kv_events,
            );
            let scheduler_start = Instant::now();
            let mut deferred_commands = VecDeque::new();

            loop {
                if !receive_until_schedulable(
                    &mut core,
                    &mut request_rx,
                    &mut command_rx,
                    &publisher,
                    &scheduler_start,
                    &cancel_token_clone,
                    controls_enabled,
                )
                .await
                {
                    break;
                }

                let iteration_start = Instant::now();
                let metrics_before = core.mocker_metrics();
                let pass = core.execute_pass_internal(None, 0.0);
                let mut pending = publisher.capture_pass(pass);
                publisher.publish_pass_start(&mut pending);
                let total_time = std::time::Duration::from_secs_f64(pending.end_ms() / 1000.0);
                let zero_progress =
                    total_time.is_zero() && !pending.made_progress_since(&metrics_before);
                if total_time > std::time::Duration::ZERO {
                    let deadline = iteration_start + total_time;
                    if controls_enabled {
                        if !wait_for_pass_boundary(
                            &mut core,
                            &mut command_rx,
                            &mut deferred_commands,
                            &publisher,
                            &scheduler_start,
                            &cancel_token_clone,
                            deadline,
                        )
                        .await
                        {
                            break;
                        }
                    } else {
                        sleep_until_precise(deadline).await;
                    }
                }
                publisher.publish_pass(&mut core, pending).await;
                if controls_enabled {
                    let mut command_processed = false;
                    while let Some(command) = deferred_commands.pop_front() {
                        command_processed = true;
                        publisher
                            .apply_command(
                                &mut core,
                                command,
                                true,
                                scheduler_elapsed_ms(&scheduler_start),
                            )
                            .await;
                    }
                    while let Ok(command) = command_rx.try_recv() {
                        command_processed = true;
                        publisher
                            .apply_command(
                                &mut core,
                                command,
                                true,
                                scheduler_elapsed_ms(&scheduler_start),
                            )
                            .await;
                    }
                    let retry_progress = publisher
                        .retry_destinations(&mut core, scheduler_elapsed_ms(&scheduler_start))
                        .await;
                    if zero_progress
                        && !command_processed
                        && !retry_progress
                        && !wait_for_progress_wake(
                            &mut core,
                            &mut request_rx,
                            &mut command_rx,
                            &publisher,
                            &scheduler_start,
                            &cancel_token_clone,
                            true,
                        )
                        .await
                    {
                        break;
                    }
                } else if zero_progress
                    && !wait_for_progress_wake(
                        &mut core,
                        &mut request_rx,
                        &mut command_rx,
                        &publisher,
                        &scheduler_start,
                        &cancel_token_clone,
                        false,
                    )
                    .await
                {
                    break;
                }
            }
        });

        Self {
            request_tx,
            command_tx,
            lifecycle_rx: Arc::new(Mutex::new(Some(lifecycle_rx))),
            metrics_rx,
            _cancel_guard: cancel_guard,
        }
    }
}

impl SchedulerHandle for SglangScheduler {
    fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request);
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.request_tx.clone()
    }

    fn metrics_receiver(&self) -> tokio::sync::watch::Receiver<MockerMetrics> {
        self.metrics_rx.clone()
    }

    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope> {
        self.command_tx.clone()
    }

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>> {
        self.lifecycle_rx
            .lock()
            .expect("scheduler lifecycle receiver mutex poisoned")
            .take()
    }
}

#[allow(clippy::too_many_arguments)]
async fn receive_until_schedulable(
    core: &mut SglangCore,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    controls_enabled: bool,
) -> bool {
    if !controls_enabled {
        if cancel_token.is_cancelled() {
            return false;
        }
        if core.is_empty() {
            tokio::select! {
                biased;
                _ = cancel_token.cancelled() => return false,
                request = request_rx.recv() => {
                    let Some(request) = request else {
                        return false;
                    };
                    core.receive(request);
                }
            }
        }
        while let Ok(request) = request_rx.try_recv() {
            core.receive(request);
        }
        return true;
    }

    while core.is_empty() {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return false,
            command = command_rx.recv() => {
                let Some(command) = command else {
                    return false;
                };
                publisher
                    .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                    .await;
            }
            result = request_rx.recv() => {
                let Some(request) = result else {
                    return false;
                };
                core.receive(request);
            }
        }
    }

    while let Ok(command) = command_rx.try_recv() {
        publisher
            .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
            .await;
    }
    while let Ok(request) = request_rx.try_recv() {
        core.receive(request);
    }

    true
}

#[allow(clippy::too_many_arguments)]
async fn wait_for_pass_boundary(
    core: &mut SglangCore,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    deferred_commands: &mut VecDeque<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    deadline: Instant,
) -> bool {
    let sleep = sleep_until_precise(deadline);
    tokio::pin!(sleep);
    loop {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return false,
            _ = &mut sleep => return true,
            command = command_rx.recv() => {
                let Some(command) = command else {
                    return false;
                };
                if command_can_apply_during_pass(&command.command) {
                    publisher
                        .apply_command(
                            core,
                            command,
                            false,
                            scheduler_elapsed_ms(scheduler_start),
                        )
                        .await;
                } else {
                    deferred_commands.push_back(command);
                    return tokio::select! {
                        biased;
                        _ = cancel_token.cancelled() => false,
                        _ = &mut sleep => true,
                    };
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn wait_for_progress_wake(
    core: &mut SglangCore,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    controls_enabled: bool,
) -> bool {
    if controls_enabled {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => false,
            command = command_rx.recv() => {
                let Some(command) = command else {
                    return false;
                };
                publisher
                    .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                    .await;
                true
            }
            request = request_rx.recv() => {
                let Some(request) = request else {
                    return false;
                };
                core.receive(request);
                true
            }
        }
    } else {
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => false,
            request = request_rx.recv() => {
                let Some(request) = request else {
                    return false;
                };
                core.receive(request);
                true
            }
        }
    }
}

fn scheduler_elapsed_ms(scheduler_start: &Instant) -> f64 {
    scheduler_start.elapsed().as_secs_f64() * 1000.0
}

fn command_can_apply_during_pass(command: &SchedulerCommand) -> bool {
    matches!(
        command,
        SchedulerCommand::CancelRequest { .. }
            | SchedulerCommand::SubmitHandoffPrefill { .. }
            | SchedulerCommand::ReserveDestination { .. }
    )
}

impl LiveBoundaryCore for SglangCore {
    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        _now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.apply_command_effects(command, allow_destination_admission)
    }

    fn retry_live_destinations(&mut self, _now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        self.retry_pending_destinations()
    }

    fn live_metrics(&self) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn pass_boundary_metrics(&self, mut pass_metrics: MockerMetrics) -> MockerMetrics {
        let current = self.mocker_metrics();
        pass_metrics.active_decode_blocks = current.active_decode_blocks;
        pass_metrics.gpu_cache_usage_perc = current.gpu_cache_usage_perc;
        pass_metrics.running_requests = current.running_requests;
        pass_metrics.waiting_requests = current.waiting_requests;
        pass_metrics
    }
}
