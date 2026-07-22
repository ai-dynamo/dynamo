// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::common::utils::sleep_until_precise;
use crate::scheduler::kv_event_sink::{
    DeferredKvPublish, DeferredKvPublishBuffer, capture_deferred_kv_publish_sink,
    publish_deferred_fpm, publish_deferred_kv_events,
};
use crate::scheduler::vllm::MockerMetrics;
use crate::scheduler::{
    AdmissionEvent, EnginePassResult, RouterEventVisibility, SchedulerCancellationEnvelope,
    SchedulerCommand, SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerCommandResult,
    SchedulerHandle, SchedulerLifecycleEvent, handoff_channel_capacity,
};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PublishedEffect {
    Admissions,
    Kv,
    Fpm,
    Outputs,
    Accounting,
    Ack,
    Lifecycle,
    Metrics,
}

pub(crate) trait LiveBoundaryCore {
    fn initialize_live(&mut self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }

    fn live_is_empty(&self) -> bool;

    fn receive_live_request(&mut self, request: DirectRequest);

    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects>;

    fn retry_live_destinations(&mut self, now_ms: f64) -> Vec<SchedulerLifecycleEvent>;

    fn live_metrics(&self) -> MockerMetrics;

    fn pass_boundary_metrics(&self, pass_metrics: MockerMetrics) -> MockerMetrics;

    fn live_internal_deadline_ms(&self) -> Option<f64> {
        None
    }

    fn execute_live_pass(&mut self, scheduler_start: &Instant) -> LivePassExecution;

    fn output_delivery_failed(&mut self, _signals: Vec<OutputSignal>) {}

    #[cfg(feature = "kvbm-offload")]
    fn advance_live_offload(
        &mut self,
        _now_ms: f64,
        _allow_destination_admission: bool,
    ) -> crate::scheduler::OffloadTickEffects {
        crate::scheduler::OffloadTickEffects {
            kv_events: Vec::new(),
            lifecycle_events: Vec::new(),
        }
    }
}

pub(crate) struct LivePassExecution {
    pub(crate) pass: EnginePassResult,
    pub(crate) duration: Duration,
}

struct LiveCancelGuard(CancellationToken);

impl Drop for LiveCancelGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[derive(Clone)]
pub(crate) struct LiveSchedulerState {
    request_tx: mpsc::UnboundedSender<DirectRequest>,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    cancellation_tx: mpsc::Sender<SchedulerCancellationEnvelope>,
    lifecycle_rx: Arc<Mutex<Option<mpsc::Receiver<SchedulerLifecycleEvent>>>>,
    metrics_rx: watch::Receiver<MockerMetrics>,
    _cancel_guard: Arc<LiveCancelGuard>,
}

impl SchedulerHandle for LiveSchedulerState {
    fn receive(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request);
    }

    fn request_sender(&self) -> mpsc::UnboundedSender<DirectRequest> {
        self.request_tx.clone()
    }

    fn metrics_receiver(&self) -> watch::Receiver<MockerMetrics> {
        self.metrics_rx.clone()
    }

    fn command_sender(&self) -> mpsc::Sender<SchedulerCommandEnvelope> {
        self.command_tx.clone()
    }

    fn cancellation_sender(&self) -> mpsc::Sender<SchedulerCancellationEnvelope> {
        self.cancellation_tx.clone()
    }

    fn take_lifecycle_receiver(&mut self) -> Option<mpsc::Receiver<SchedulerLifecycleEvent>> {
        self.lifecycle_rx
            .lock()
            .expect("scheduler lifecycle receiver mutex poisoned")
            .take()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_live_scheduler<C>(
    args: MockEngineArgs,
    dp_rank: u32,
    output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    kv_event_publishers: KvEventPublishers,
    cancellation_token: Option<CancellationToken>,
    admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
    fpm_publisher: FpmPublisher,
    make_core: impl FnOnce(MockEngineArgs, u32, KvEventPublishers) -> C + Send + 'static,
) -> LiveSchedulerState
where
    C: LiveBoundaryCore + Send + 'static,
{
    let controls_enabled = args.is_prefill() || args.is_decode();
    let (request_tx, request_rx) = mpsc::unbounded_channel();
    let control_capacity = handoff_channel_capacity(&args);
    let (command_tx, command_rx) = mpsc::channel(control_capacity);
    let (cancellation_tx, cancellation_rx) = mpsc::channel(control_capacity);
    let (lifecycle_tx, lifecycle_rx) = mpsc::channel(control_capacity);
    let initial_metrics = MockerMetrics::new(dp_rank, 0, args.num_gpu_blocks as u64);
    let (metrics_tx, metrics_rx) = watch::channel(initial_metrics);
    let cancel_token = cancellation_token.unwrap_or_default();
    let actor_cancel_token = cancel_token.clone();
    let cancel_guard = Arc::new(LiveCancelGuard(cancel_token));

    tokio::spawn(async move {
        let (deferred_kv_events, buffering_publishers) = capture_deferred_kv_publish_sink(
            !kv_event_publishers.is_empty(),
            kv_event_publishers.raw_enabled(),
        );
        let mut core = make_core(args, dp_rank, buffering_publishers);
        let publisher = LiveEffectsPublisher::new(
            output_tx,
            admission_tx,
            lifecycle_tx,
            metrics_tx,
            kv_event_publishers,
            fpm_publisher,
            deferred_kv_events,
        );
        core.initialize_live().await;
        run_live_scheduler(
            &mut core,
            request_rx,
            command_rx,
            cancellation_rx,
            publisher,
            actor_cancel_token,
            controls_enabled,
        )
        .await;
    });

    LiveSchedulerState {
        request_tx,
        command_tx,
        cancellation_tx,
        lifecycle_rx: Arc::new(Mutex::new(Some(lifecycle_rx))),
        metrics_rx,
        _cancel_guard: cancel_guard,
    }
}

async fn run_live_scheduler<C: LiveBoundaryCore>(
    core: &mut C,
    mut request_rx: mpsc::UnboundedReceiver<DirectRequest>,
    mut command_rx: mpsc::Receiver<SchedulerCommandEnvelope>,
    mut cancellation_rx: mpsc::Receiver<SchedulerCancellationEnvelope>,
    publisher: LiveEffectsPublisher,
    cancel_token: CancellationToken,
    controls_enabled: bool,
) {
    let scheduler_start = Instant::now();
    let mut deferred_commands = VecDeque::new();

    loop {
        // Productive zero-duration passes may never enter one of the
        // cancellation-aware waits below.
        if cancel_token.is_cancelled() {
            break;
        }
        if !receive_until_live_schedulable(
            core,
            &mut request_rx,
            &mut command_rx,
            &mut cancellation_rx,
            &publisher,
            &scheduler_start,
            &cancel_token,
            controls_enabled,
        )
        .await
        {
            break;
        }

        let iteration_start = Instant::now();
        let metrics_before = core.live_metrics();
        let execution = core.execute_live_pass(&scheduler_start);
        let mut pending = publisher.capture_pass(execution.pass);
        let zero_progress =
            execution.duration.is_zero() && !pending.made_progress_since(&metrics_before);
        publisher.publish_pass_start(&mut pending);
        if execution.duration > Duration::ZERO {
            let deadline = iteration_start + execution.duration;
            if !wait_for_live_pass_boundary(
                core,
                &mut command_rx,
                &mut cancellation_rx,
                &mut deferred_commands,
                &mut pending,
                &publisher,
                &scheduler_start,
                &cancel_token,
                deadline,
                controls_enabled,
            )
            .await
            {
                break;
            }
        }
        publisher.publish_pass(core, pending).await;
        let control_progress = apply_live_post_pass_controls(
            core,
            &mut command_rx,
            &mut cancellation_rx,
            &mut deferred_commands,
            &publisher,
            &scheduler_start,
            controls_enabled,
        )
        .await;
        if zero_progress
            && !control_progress
            && !wait_for_live_progress(
                core,
                &mut request_rx,
                &mut command_rx,
                &mut cancellation_rx,
                &publisher,
                &scheduler_start,
                &cancel_token,
                controls_enabled,
            )
            .await
        {
            break;
        }
        if execution.duration.is_zero() && !zero_progress {
            tokio::task::coop::consume_budget().await;
        }
    }
}

/// Owns every effect that becomes externally visible at a live scheduler
/// boundary. Pass effects are captured before modeled sleep so an admissible
/// mid-pass command cannot publish or consume state derived from that pass.
pub(crate) struct LiveEffectsPublisher {
    output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
    lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
    metrics_tx: watch::Sender<MockerMetrics>,
    kv_event_publishers: KvEventPublishers,
    fpm_publisher: FpmPublisher,
    captured_kv_events: DeferredKvPublishBuffer,
    #[cfg(test)]
    publication_log: Option<std::sync::Arc<std::sync::Mutex<Vec<PublishedEffect>>>>,
}

pub(crate) struct PendingLivePass {
    pass: EnginePassResult,
    kv_events: Vec<DeferredKvPublish>,
    admissions_published: bool,
    pass_start_kv_published: bool,
}

impl PendingLivePass {
    pub(crate) fn made_progress_since(&self, metrics_before: &MockerMetrics) -> bool {
        self.pass.completed_requests > 0
            || !self.pass.admissions.is_empty()
            || !self.pass.output_signals.is_empty()
            || !self.pass.lifecycle_events.is_empty()
            || !self.pass.kv_events.is_empty()
            || !self.kv_events.is_empty()
            || self.pass.mocker_metrics != *metrics_before
    }

    pub(crate) fn suppress_request_outputs(&mut self, request_id: Uuid) {
        self.pass
            .output_signals
            .retain(|signal| signal.uuid != request_id);
        self.pass.completed_requests = self
            .pass
            .output_signals
            .iter()
            .filter(|signal| signal.completed)
            .count();
        let (output_tokens, decode_forwards) =
            crate::scheduler::accept_length_sample(&self.pass.output_signals);
        self.pass.accept_length_output_tokens = output_tokens;
        self.pass.accept_length_decode_forwards = decode_forwards;
    }
}

#[allow(clippy::too_many_arguments)]
async fn receive_until_live_schedulable<C: LiveBoundaryCore>(
    core: &mut C,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    cancellation_rx: &mut mpsc::Receiver<SchedulerCancellationEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    controls_enabled: bool,
) -> bool {
    while core.live_is_empty() {
        let internal_deadline_ms = core.live_internal_deadline_ms();
        let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
        tokio::pin!(internal_deadline);
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return false,
            cancellation = cancellation_rx.recv() => {
                let Some(cancellation) = cancellation else {
                    return false;
                };
                let _ = publisher
                    .apply_cancellation(
                        core,
                        cancellation,
                        true,
                        scheduler_elapsed_ms(scheduler_start),
                    )
                    .await;
            }
            command = command_rx.recv(), if controls_enabled => {
                let Some(command) = command else {
                    return false;
                };
                publisher
                    .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                    .await;
            }
            request = request_rx.recv() => {
                let Some(request) = request else {
                    return false;
                };
                core.receive_live_request(request);
            }
            _ = &mut internal_deadline, if controls_enabled && internal_deadline_ms.is_some() => {
                #[cfg(feature = "kvbm-offload")]
                {
                    let now_ms = scheduler_elapsed_ms(scheduler_start)
                        .max(internal_deadline_ms.expect("armed internal deadline"));
                    publisher.advance_offload(core, now_ms, true).await;
                }
            }
        }
    }

    while let Ok(cancellation) = cancellation_rx.try_recv() {
        let _ = publisher
            .apply_cancellation(
                core,
                cancellation,
                true,
                scheduler_elapsed_ms(scheduler_start),
            )
            .await;
    }
    if controls_enabled {
        while let Ok(command) = command_rx.try_recv() {
            publisher
                .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                .await;
        }
    }
    while let Ok(request) = request_rx.try_recv() {
        core.receive_live_request(request);
    }

    true
}

#[allow(clippy::too_many_arguments)]
async fn wait_for_live_pass_boundary<C: LiveBoundaryCore>(
    core: &mut C,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    cancellation_rx: &mut mpsc::Receiver<SchedulerCancellationEnvelope>,
    deferred_commands: &mut VecDeque<SchedulerCommandEnvelope>,
    pending: &mut PendingLivePass,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    deadline: Instant,
    controls_enabled: bool,
) -> bool {
    let sleep = sleep_until_precise(deadline);
    tokio::pin!(sleep);
    let mut accept_commands = true;
    loop {
        let internal_deadline_ms = core.live_internal_deadline_ms();
        let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
        tokio::pin!(internal_deadline);
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return false,
            _ = &mut sleep => return true,
            cancellation = cancellation_rx.recv() => {
                let Some(cancellation) = cancellation else {
                    return false;
                };
                let request_id = cancellation.request_id;
                let discard_pending_output = cancellation.discard_pending_output;
                let outcome = publisher
                    .apply_cancellation(
                        core,
                        cancellation,
                        false,
                        scheduler_elapsed_ms(scheduler_start),
                    )
                    .await;
                if discard_pending_output || outcome != Some(SchedulerCommandResult::Noop) {
                    pending.suppress_request_outputs(request_id);
                }
            }
            _ = &mut internal_deadline, if controls_enabled && internal_deadline_ms.is_some() => {
                #[cfg(feature = "kvbm-offload")]
                {
                    let now_ms = scheduler_elapsed_ms(scheduler_start)
                        .max(internal_deadline_ms.expect("armed internal deadline"));
                    publisher.advance_offload(core, now_ms, false).await;
                    debug_assert!(
                        core.live_internal_deadline_ms().is_none_or(|next| next > now_ms),
                        "internal progress left an already-due deadline armed"
                    );
                }
            }
            command = command_rx.recv(), if controls_enabled && accept_commands => {
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
                    accept_commands = false;
                }
            }
        }
    }
}

async fn apply_live_post_pass_controls<C: LiveBoundaryCore>(
    core: &mut C,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    cancellation_rx: &mut mpsc::Receiver<SchedulerCancellationEnvelope>,
    deferred_commands: &mut VecDeque<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    controls_enabled: bool,
) -> bool {
    let mut made_progress = false;
    while let Ok(cancellation) = cancellation_rx.try_recv() {
        made_progress = true;
        let _ = publisher
            .apply_cancellation(
                core,
                cancellation,
                true,
                scheduler_elapsed_ms(scheduler_start),
            )
            .await;
    }
    if controls_enabled {
        while let Some(command) = deferred_commands.pop_front() {
            made_progress = true;
            publisher
                .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                .await;
        }
        while let Ok(command) = command_rx.try_recv() {
            made_progress = true;
            publisher
                .apply_command(core, command, true, scheduler_elapsed_ms(scheduler_start))
                .await;
        }
        publisher
            .retry_destinations(core, scheduler_elapsed_ms(scheduler_start))
            .await
            || made_progress
    } else {
        made_progress
    }
}

#[allow(clippy::too_many_arguments)]
async fn wait_for_live_progress<C: LiveBoundaryCore>(
    core: &mut C,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    cancellation_rx: &mut mpsc::Receiver<SchedulerCancellationEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    controls_enabled: bool,
) -> bool {
    let internal_deadline_ms = core.live_internal_deadline_ms();
    let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
    tokio::pin!(internal_deadline);
    tokio::select! {
        biased;
        _ = cancel_token.cancelled() => false,
        cancellation = cancellation_rx.recv() => {
            let Some(cancellation) = cancellation else {
                return false;
            };
            let _ = publisher
                .apply_cancellation(
                    core,
                    cancellation,
                    true,
                    scheduler_elapsed_ms(scheduler_start),
                )
                .await;
            true
        }
        command = command_rx.recv(), if controls_enabled => {
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
            core.receive_live_request(request);
            true
        }
        _ = &mut internal_deadline => true,
    }
}

async fn wait_for_internal_deadline(scheduler_start: &Instant, deadline_ms: Option<f64>) {
    let Some(deadline_ms) = deadline_ms else {
        std::future::pending::<()>().await;
        return;
    };
    let deadline = *scheduler_start + Duration::from_secs_f64(deadline_ms.max(0.0) / 1000.0);
    let wake_at = if deadline <= Instant::now() {
        Instant::now() + Duration::from_millis(1)
    } else {
        deadline
    };
    sleep_until_precise(wake_at).await;
}

fn scheduler_elapsed_ms(scheduler_start: &Instant) -> f64 {
    scheduler_start.elapsed().as_secs_f64() * 1000.0
}

fn command_can_apply_during_pass(command: &SchedulerCommand) -> bool {
    matches!(
        command,
        SchedulerCommand::SubmitHandoffPrefill { .. } | SchedulerCommand::ReserveDestination { .. }
    )
}

impl LiveEffectsPublisher {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
        admission_tx: Option<mpsc::UnboundedSender<AdmissionEvent>>,
        lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
        metrics_tx: watch::Sender<MockerMetrics>,
        kv_event_publishers: KvEventPublishers,
        fpm_publisher: FpmPublisher,
        captured_kv_events: DeferredKvPublishBuffer,
    ) -> Self {
        Self {
            output_tx,
            admission_tx,
            lifecycle_tx,
            metrics_tx,
            kv_event_publishers,
            fpm_publisher,
            captured_kv_events,
            #[cfg(test)]
            publication_log: None,
        }
    }

    #[cfg(test)]
    fn with_publication_log(
        mut self,
        log: std::sync::Arc<std::sync::Mutex<Vec<PublishedEffect>>>,
    ) -> Self {
        self.publication_log = Some(log);
        self
    }

    pub(crate) fn capture_pass(&self, pass: EnginePassResult) -> PendingLivePass {
        PendingLivePass {
            pass,
            kv_events: self.captured_kv_events.drain(),
            admissions_published: false,
            pass_start_kv_published: false,
        }
    }

    /// Publish effects whose scheduler contract makes them visible before the
    /// modeled GPU pass runs. Outputs and lifecycle effects remain at pass end.
    pub(crate) fn publish_pass_start(&self, pending: &mut PendingLivePass) {
        self.publish_admissions(&pending.pass.admissions);
        pending.admissions_published = true;
        if pending.pass.router_event_visibility == RouterEventVisibility::PassStart {
            self.publish_router_effects(std::mem::take(&mut pending.kv_events), None);
            pending.pass_start_kv_published = true;
        }
    }

    /// Publishes a completed pass as one ordered transaction:
    /// admissions/accounting, KV/FPM, outputs, lifecycle, then metrics.
    pub(crate) async fn publish_pass<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        mut pending: PendingLivePass,
    ) {
        if !pending.admissions_published {
            self.publish_admissions(&pending.pass.admissions);
        }

        // Mid-pass command effects remain part of this pass transaction and
        // become visible at pass end, even when pass-start effects are already
        // published.
        let midpass_kv_events = self.captured_kv_events.drain();
        if pending.pass_start_kv_published {
            self.publish_router_effects(midpass_kv_events, pending.pass.fpm.take());
        } else {
            pending.kv_events.extend(midpass_kv_events);
            self.publish_router_effects(pending.kv_events, pending.pass.fpm.take());
        }

        #[cfg(debug_assertions)]
        {
            let completed_signals = pending
                .pass
                .output_signals
                .iter()
                .filter(|signal| signal.completed)
                .count();
            let accept_length =
                crate::scheduler::accept_length_sample(&pending.pass.output_signals);
            debug_assert_eq!(pending.pass.completed_requests, completed_signals);
            debug_assert_eq!(
                (
                    pending.pass.accept_length_output_tokens,
                    pending.pass.accept_length_decode_forwards,
                ),
                accept_length
            );
        }
        self.publish_outputs(core, pending.pass.output_signals);
        // Live completion/accept accounting is carried by the admission and
        // output channels; the scalar replay counters have no separate live
        // consumer, but are committed at this same boundary.
        self.record(PublishedEffect::Accounting);
        // vLLM may release request-owned blocks when its output receiver has
        // disappeared. Those cleanup events belong to this same boundary.
        self.publish_router_effects(self.captured_kv_events.drain(), None);
        self.publish_lifecycle(pending.pass.lifecycle_events).await;
        let metrics = core.pass_boundary_metrics(pending.pass.mocker_metrics);
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(metrics);
    }

    pub(crate) async fn apply_command<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        envelope: SchedulerCommandEnvelope,
        allow_destination_admission: bool,
        now_ms: f64,
    ) {
        self.apply_command_inner(core, envelope, allow_destination_admission, now_ms)
            .await;
    }

    pub(crate) async fn apply_cancellation<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        cancellation: SchedulerCancellationEnvelope,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> Option<SchedulerCommandResult> {
        self.apply_command_inner(
            core,
            cancellation.into(),
            allow_destination_admission,
            now_ms,
        )
        .await
    }

    async fn apply_command_inner<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        envelope: SchedulerCommandEnvelope,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> Option<SchedulerCommandResult> {
        let SchedulerCommandEnvelope { command, reply } = envelope;
        let result = core.apply_live_command(command, allow_destination_admission, now_ms);
        match result {
            Ok(mut effects) => {
                let lifecycle_events = std::mem::take(&mut effects.lifecycle_events);
                let command_result = effects.result;
                if allow_destination_admission {
                    self.publish_router_effects(self.captured_kv_events.drain(), None);
                } else {
                    assert!(
                        lifecycle_events.is_empty(),
                        "mid-pass scheduler command produced lifecycle effects"
                    );
                }
                self.record(PublishedEffect::Ack);
                let _ = reply.send(Ok(effects));
                if allow_destination_admission {
                    self.publish_lifecycle(lifecycle_events).await;
                    self.record(PublishedEffect::Metrics);
                    let _ = self.metrics_tx.send(core.live_metrics());
                }
                Some(command_result)
            }
            Err(error) => {
                self.record(PublishedEffect::Ack);
                let _ = reply.send(Err(error));
                None
            }
        }
    }

    pub(crate) async fn retry_destinations<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        now_ms: f64,
    ) -> bool {
        let lifecycle_events = core.retry_live_destinations(now_ms);
        let kv_events = self.captured_kv_events.drain();
        let made_progress = !lifecycle_events.is_empty() || !kv_events.is_empty();
        self.publish_router_effects(kv_events, None);
        self.publish_lifecycle(lifecycle_events).await;
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(core.live_metrics());
        made_progress
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) async fn advance_offload<C: LiveBoundaryCore>(
        &self,
        core: &mut C,
        now_ms: f64,
        allow_destination_admission: bool,
    ) -> bool {
        let effects = core.advance_live_offload(now_ms, allow_destination_admission);
        assert!(
            effects.kv_events.is_empty(),
            "live offload progress unexpectedly used offline KV capture"
        );
        if !allow_destination_admission {
            assert!(
                effects.lifecycle_events.is_empty(),
                "busy offload progress admitted a destination"
            );
        }
        let kv_events = self.captured_kv_events.drain();
        let made_progress = !kv_events.is_empty() || !effects.lifecycle_events.is_empty();
        self.publish_router_effects(kv_events, None);
        self.publish_lifecycle(effects.lifecycle_events).await;
        self.record(PublishedEffect::Metrics);
        let _ = self.metrics_tx.send(core.live_metrics());
        made_progress
    }

    fn publish_admissions(&self, admissions: &[AdmissionEvent]) {
        let Some(tx) = self.admission_tx.as_ref() else {
            return;
        };
        if !admissions.is_empty() {
            self.record(PublishedEffect::Admissions);
        }
        for admission in admissions {
            let _ = tx.send(admission.clone());
        }
    }

    fn publish_router_effects(
        &self,
        kv_events: Vec<DeferredKvPublish>,
        fpm: Option<crate::common::protocols::ForwardPassSnapshot>,
    ) {
        if !kv_events.is_empty() {
            self.record(PublishedEffect::Kv);
        }
        publish_deferred_kv_events(&self.kv_event_publishers, kv_events);
        if let Some(fpm) = fpm {
            self.record(PublishedEffect::Fpm);
            publish_deferred_fpm(&self.fpm_publisher, vec![fpm]);
        }
    }

    fn publish_outputs<C: LiveBoundaryCore>(&self, core: &mut C, signals: Vec<OutputSignal>) {
        let Some(tx) = self.output_tx.as_ref() else {
            return;
        };
        if signals.is_empty() {
            return;
        }
        self.record(PublishedEffect::Outputs);
        if let Err(error) = tx.send(signals) {
            core.output_delivery_failed(error.0);
        }
    }

    async fn publish_lifecycle(&self, events: Vec<SchedulerLifecycleEvent>) {
        if !events.is_empty() {
            self.record(PublishedEffect::Lifecycle);
        }
        for event in events {
            if self.lifecycle_tx.send(event).await.is_err() {
                break;
            }
        }
    }

    fn record(&self, effect: PublishedEffect) {
        #[cfg(test)]
        if let Some(log) = &self.publication_log {
            log.lock().unwrap().push(effect);
        }
        #[cfg(not(test))]
        let _ = effect;
    }
}

#[cfg(test)]
mod tests;
