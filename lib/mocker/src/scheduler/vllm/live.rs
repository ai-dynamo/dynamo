// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::common::utils::sleep_until_precise;
use crate::scheduler::{
    AdmissionEvent, LiveBoundaryCore, LiveEffectsPublisher, SchedulerCommand,
    SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerHandle, SchedulerLifecycleEvent,
    capture_deferred_kv_publish_sink, handoff_channel_capacity,
};

use super::core::VllmCore;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct MockerMetrics {
    pub dp_rank: dynamo_kv_router::protocols::DpRank,
    pub active_decode_blocks: u64,
    pub total_blocks: u64,
    pub gpu_cache_usage_perc: f64,
    pub running_requests: u64,
    pub waiting_requests: u64,
    pub vllm_preemptions_total: u64,
    pub sglang_cache_hit_tokens: u64,
    pub sglang_cache_total_tokens: u64,
}

impl MockerMetrics {
    pub fn new(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
    ) -> Self {
        Self::from_parts(dp_rank, active_decode_blocks, total_blocks, 0, 0, 0, 0, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        dp_rank: dynamo_kv_router::protocols::DpRank,
        active_decode_blocks: u64,
        total_blocks: u64,
        running_requests: u64,
        waiting_requests: u64,
        vllm_preemptions_total: u64,
        sglang_cache_hit_tokens: u64,
        sglang_cache_total_tokens: u64,
    ) -> Self {
        let gpu_cache_usage_perc = if total_blocks == 0 {
            0.0
        } else {
            active_decode_blocks as f64 / total_blocks as f64
        };
        Self {
            dp_rank,
            active_decode_blocks,
            total_blocks,
            gpu_cache_usage_perc,
            running_requests,
            waiting_requests,
            vllm_preemptions_total,
            sglang_cache_hit_tokens,
            sglang_cache_total_tokens,
        }
    }
}

#[derive(Clone)]
pub struct Scheduler {
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

impl Scheduler {
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
        let (metrics_tx, metrics_rx) = tokio::sync::watch::channel(initial_metrics);

        let cancel_token = cancellation_token.unwrap_or_default();
        let cancel_token_clone = cancel_token.clone();
        let cancel_guard = Arc::new(CancelGuard(cancel_token));
        let controls_enabled = args.is_prefill() || args.is_decode();

        tokio::spawn(async move {
            let (deferred_kv_events, buffering_publishers) = capture_deferred_kv_publish_sink(
                !kv_event_publishers.is_empty(),
                kv_event_publishers.raw_enabled(),
            );
            let mut core = VllmCore::new_with_sink(args, dp_rank, buffering_publishers);
            let publisher = LiveEffectsPublisher::new(
                output_tx,
                admission_tx,
                lifecycle_tx,
                metrics_tx,
                kv_event_publishers,
                fpm_publisher,
                deferred_kv_events,
            );
            #[cfg(feature = "kvbm-offload")]
            if let Err(e) = core.init_offload_live().await {
                tracing::error!("kvbm-offload live init failed: {e}");
            }
            // Wall-clock origin for this scheduler's simulated time. Drives
            // `engine.tick(now_ms)` so the PS bandwidth models advance
            // in real time across passes.
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
                let now_ms = scheduler_start.elapsed().as_secs_f64() * 1000.0;
                let metrics_before = core.mocker_metrics();
                let pass = core.execute_pass_internal(None, now_ms, None);
                let mut pending = publisher.capture_pass(pass);
                let total_time = std::time::Duration::from_secs_f64(
                    (pending.end_ms() - now_ms).max(0.0) / 1000.0,
                );
                let zero_progress =
                    total_time.is_zero() && !pending.made_progress_since(&metrics_before);
                publisher.publish_pass_start(&mut pending);
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

impl SchedulerHandle for Scheduler {
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
    core: &mut VllmCore,
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
        #[cfg(feature = "kvbm-offload")]
        let internal_deadline_ms = core.earliest_offload_deadline();
        #[cfg(not(feature = "kvbm-offload"))]
        let internal_deadline_ms = None;
        let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
        tokio::pin!(internal_deadline);
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
            _ = &mut internal_deadline, if internal_deadline_ms.is_some() => {
                #[cfg(feature = "kvbm-offload")]
                {
                    let now_ms = scheduler_elapsed_ms(scheduler_start)
                        .max(internal_deadline_ms.expect("armed offload deadline"));
                    publisher.advance_offload(core, now_ms, true).await;
                }
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
    core: &mut VllmCore,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    deferred_commands: &mut VecDeque<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    deadline: Instant,
) -> bool {
    let sleep = sleep_until_precise(deadline);
    tokio::pin!(sleep);
    let mut accept_commands = true;
    loop {
        #[cfg(feature = "kvbm-offload")]
        let internal_deadline_ms = core.earliest_offload_deadline();
        #[cfg(not(feature = "kvbm-offload"))]
        let internal_deadline_ms = None;
        let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
        tokio::pin!(internal_deadline);
        tokio::select! {
            biased;
            _ = cancel_token.cancelled() => return false,
            _ = &mut sleep => return true,
            _ = &mut internal_deadline, if internal_deadline_ms.is_some() => {
                #[cfg(feature = "kvbm-offload")]
                {
                    let now_ms = scheduler_elapsed_ms(scheduler_start)
                        .max(internal_deadline_ms.expect("armed offload deadline"));
                    publisher.advance_offload(core, now_ms, false).await;
                    debug_assert!(
                        core.earliest_offload_deadline()
                            .is_none_or(|next| next > now_ms),
                        "offload tick left an already-due deadline armed"
                    );
                }
            }
            command = command_rx.recv(), if accept_commands => {
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

#[allow(clippy::too_many_arguments)]
async fn wait_for_progress_wake(
    core: &mut VllmCore,
    request_rx: &mut mpsc::UnboundedReceiver<DirectRequest>,
    command_rx: &mut mpsc::Receiver<SchedulerCommandEnvelope>,
    publisher: &LiveEffectsPublisher,
    scheduler_start: &Instant,
    cancel_token: &CancellationToken,
    controls_enabled: bool,
) -> bool {
    #[cfg(feature = "kvbm-offload")]
    let internal_deadline_ms = core.earliest_offload_deadline();
    #[cfg(not(feature = "kvbm-offload"))]
    let internal_deadline_ms = None;

    let internal_deadline = wait_for_internal_deadline(scheduler_start, internal_deadline_ms);
    tokio::pin!(internal_deadline);
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
            _ = &mut internal_deadline => true,
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
            _ = &mut internal_deadline => true,
        }
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

impl LiveBoundaryCore for VllmCore {
    fn apply_live_command(
        &mut self,
        command: SchedulerCommand,
        allow_destination_admission: bool,
        now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.apply_command_effects_at(command, allow_destination_admission, Some(now_ms))
    }

    fn retry_live_destinations(&mut self, now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        self.retry_pending_destinations_at(Some(now_ms))
    }

    fn live_metrics(&self) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn pass_boundary_metrics(&self, _pass_metrics: MockerMetrics) -> MockerMetrics {
        self.mocker_metrics()
    }

    fn output_delivery_failed(&mut self, signals: Vec<OutputSignal>) {
        for signal in signals {
            self.drop_request(signal.uuid);
        }
    }

    #[cfg(feature = "kvbm-offload")]
    fn advance_live_offload(
        &mut self,
        now_ms: f64,
        allow_destination_admission: bool,
    ) -> crate::scheduler::OffloadTickEffects {
        if allow_destination_admission {
            self.tick_offload_only(now_ms)
        } else {
            self.tick_offload_transport_only(now_ms)
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "kvbm-offload")]
    use super::*;

    #[cfg(feature = "kvbm-offload")]
    use crate::common::protocols::KvCacheEventSink;
    #[cfg(feature = "kvbm-offload")]
    use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier};

    #[cfg(feature = "kvbm-offload")]
    #[derive(Default)]
    struct CapturingKvSink {
        events: Mutex<Vec<(StorageTier, KvCacheEvent)>>,
    }

    #[cfg(feature = "kvbm-offload")]
    impl KvCacheEventSink for CapturingKvSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push((StorageTier::Device, event));
            Ok(())
        }

        fn publish_with_storage_tier(
            &self,
            event: KvCacheEvent,
            storage_tier: StorageTier,
        ) -> anyhow::Result<()> {
            self.events.lock().unwrap().push((storage_tier, event));
            Ok(())
        }
    }

    #[cfg(feature = "kvbm-offload")]
    #[tokio::test]
    async fn idle_destination_reservation_wakes_at_offload_deadline() {
        let args = MockEngineArgs::builder()
            .num_gpu_blocks(2)
            .block_size(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(1))
            .enable_prefix_caching(true)
            .worker_type(crate::common::protocols::WorkerType::Decode)
            .speedup_ratio(1000.0)
            .kv_bytes_per_token(Some(250_000))
            .num_g2_blocks(Some(4))
            .bandwidth_g1_to_g2_gbps(Some(1.0))
            .build()
            .unwrap();
        let sink = Arc::new(CapturingKvSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone()), None);
        let (output_tx, mut output_rx) = mpsc::unbounded_channel();
        let mut scheduler = Scheduler::new(
            args,
            0,
            Some(output_tx),
            publishers,
            None,
            FpmPublisher::default(),
        );
        let mut lifecycle_rx = scheduler.take_lifecycle_receiver().unwrap();

        scheduler.receive(DirectRequest {
            tokens: vec![1; 4],
            max_output_tokens: 1,
            uuid: Some(uuid::Uuid::from_u128(1)),
            ..Default::default()
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let seed = output_rx
                    .recv()
                    .await
                    .expect("output channel should stay open");
                if seed.iter().any(|signal| signal.completed) {
                    break;
                }
            }
        })
        .await
        .expect("seed request should complete");
        sink.events.lock().unwrap().clear();

        let source_handoff_id = crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(10));
        let source_request_id = uuid::Uuid::from_u128(11);
        let (source_reply, source_reply_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::SubmitHandoffPrefill {
                    handoff_id: source_handoff_id,
                    request: DirectRequest {
                        tokens: vec![3; 3],
                        max_output_tokens: 1,
                        uuid: Some(source_request_id),
                        ..Default::default()
                    },
                },
                reply: source_reply,
            })
            .await
            .unwrap();
        let submitted = source_reply_rx.await.unwrap().unwrap();
        assert!(matches!(
            submitted.result,
            crate::scheduler::SchedulerCommandResult::Submitted(observed)
                if observed == source_request_id
        ));
        let held = tokio::time::timeout(Duration::from_secs(1), lifecycle_rx.recv())
            .await
            .expect("source hold should complete")
            .expect("lifecycle channel should stay open");
        assert!(matches!(
            held,
            SchedulerLifecycleEvent::SourceHeld {
                handoff_id: observed,
                request_id: observed_request,
                ..
            } if observed == source_handoff_id && observed_request == source_request_id
        ));

        let handoff_id = crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(2));
        let request_id = uuid::Uuid::from_u128(3);
        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request: DirectRequest {
                        tokens: vec![2; 4],
                        max_output_tokens: 1,
                        uuid: Some(request_id),
                        ..Default::default()
                    },
                },
                reply,
            })
            .await
            .unwrap();
        let accepted = reply_rx.await.unwrap().unwrap();
        assert!(matches!(
            accepted.result,
            crate::scheduler::SchedulerCommandResult::DestinationAccepted {
                request_id: observed,
            } if observed == request_id
        ));
        assert!(accepted.lifecycle_events.is_empty());

        let reserved = tokio::time::timeout(Duration::from_secs(1), lifecycle_rx.recv())
            .await
            .expect("idle offload deadline should wake the scheduler")
            .expect("lifecycle channel should stay open");
        assert!(matches!(
            reserved,
            SchedulerLifecycleEvent::DestinationReserved {
                handoff_id: observed,
                request_id: observed_request,
                ..
            } if observed == handoff_id && observed_request == request_id
        ));
        let host_stores = || {
            sink.events
                .lock()
                .unwrap()
                .iter()
                .filter(|(tier, event)| {
                    *tier == StorageTier::HostPinned
                        && matches!(&event.data, KvCacheEventData::Stored(_))
                })
                .count()
        };
        assert_eq!(host_stores(), 1);

        let (barrier_reply, barrier_rx) = tokio::sync::oneshot::channel();
        scheduler
            .command_sender()
            .send(SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelDestination {
                    handoff_id: crate::common::handoff::HandoffId::from(uuid::Uuid::from_u128(99)),
                },
                reply: barrier_reply,
            })
            .await
            .unwrap();
        let barrier = barrier_rx.await.unwrap().unwrap();
        assert_eq!(
            barrier.result,
            crate::scheduler::SchedulerCommandResult::Noop
        );
        assert!(lifecycle_rx.try_recv().is_err());
        assert_eq!(host_stores(), 1);

        for command in [
            SchedulerCommand::CancelDestination { handoff_id },
            SchedulerCommand::CancelSource {
                handoff_id: source_handoff_id,
            },
        ] {
            let (reply, reply_rx) = tokio::sync::oneshot::channel();
            scheduler
                .command_sender()
                .send(SchedulerCommandEnvelope { command, reply })
                .await
                .unwrap();
            let cleanup = reply_rx.await.unwrap().unwrap();
            assert_eq!(
                cleanup.result,
                crate::scheduler::SchedulerCommandResult::Applied
            );
        }
    }
}
