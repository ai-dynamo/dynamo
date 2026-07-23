// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::handoff::HandoffId;
use crate::common::protocols::ForwardPassSnapshot;
use crate::scheduler::SchedulerCommandResult;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

struct FakeCore {
    publishers: KvEventPublishers,
    command_result: SchedulerCommandResult,
    command_effects: bool,
    midpass_kv_effects: bool,
    execute_count: usize,
    live_pass_limit: Option<usize>,
    refill_command_tx: Option<mpsc::Sender<SchedulerCommandEnvelope>>,
    applied_command_count: usize,
}

impl FakeCore {
    fn publish_kv(&self, event_id: u64) {
        self.publishers
            .publish(
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Cleared,
                    dp_rank: 0,
                },
                None,
            )
            .unwrap();
    }
}

impl LiveBoundaryCore for FakeCore {
    fn live_is_empty(&self) -> bool {
        self.live_pass_limit.is_none()
    }

    fn receive_live_request(&mut self, _request: DirectRequest) {}

    fn execute_live_pass(&mut self, _scheduler_start: &Instant) -> LivePassExecution {
        let live_pass_limit = self
            .live_pass_limit
            .as_ref()
            .expect("boundary-only fake does not execute live passes");
        self.execute_count += 1;
        assert!(
            self.execute_count <= *live_pass_limit,
            "scheduler did not yield to externally triggered shutdown"
        );
        let mut pass = pass();
        pass.admissions.clear();
        pass.lifecycle_events.clear();
        pass.fpm = None;
        LivePassExecution {
            pass,
            duration: Duration::ZERO,
        }
    }

    fn apply_live_command(
        &mut self,
        _command: SchedulerCommand,
        _allow_destination_admission: bool,
        _now_ms: f64,
    ) -> anyhow::Result<SchedulerCommandEffects> {
        self.applied_command_count += 1;
        if let Some(command_tx) = self.refill_command_tx.take() {
            let (reply, _reply_rx) = tokio::sync::oneshot::channel();
            command_tx
                .try_send(SchedulerCommandEnvelope {
                    command: SchedulerCommand::CancelSource {
                        handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    },
                    reply,
                })
                .unwrap();
        }
        let mut effects = SchedulerCommandEffects::new(self.command_result);
        if self.command_effects || self.midpass_kv_effects {
            self.publish_kv(2);
        }
        if self.command_effects {
            effects
                .lifecycle_events
                .push(SchedulerLifecycleEvent::DestinationReserved {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                    request_id: Uuid::from_u128(3),
                    transferable_prompt_tokens: 4,
                });
        }
        Ok(effects)
    }

    fn retry_live_destinations(&mut self, _now_ms: f64) -> Vec<SchedulerLifecycleEvent> {
        Vec::new()
    }

    fn live_metrics(&self) -> MockerMetrics {
        MockerMetrics::default()
    }

    fn pass_boundary_metrics(&self, pass_metrics: MockerMetrics) -> MockerMetrics {
        pass_metrics
    }

    fn output_delivery_failed(&mut self, _signals: Vec<OutputSignal>) {
        self.publish_kv(3);
    }
}

fn pass() -> EnginePassResult {
    let request_id = Uuid::from_u128(1);
    EnginePassResult {
        end_ms: 1.0,
        completed_requests: 1,
        output_signals: vec![OutputSignal {
            uuid: request_id,
            token_id: Some(1),
            completed: true,
            rejected: false,
            handoff_delay_ms: None,
        }],
        admissions: vec![AdmissionEvent {
            uuid: request_id,
            reused_input_tokens: 0,
        }],
        lifecycle_events: vec![SchedulerLifecycleEvent::DestinationReserved {
            handoff_id: HandoffId::from(Uuid::from_u128(2)),
            request_id,
            transferable_prompt_tokens: 4,
        }],
        mocker_metrics: MockerMetrics::default(),
        router_event_visibility: RouterEventVisibility::PassEnd,
        kv_events: Vec::new(),
        fpm: Some(ForwardPassSnapshot::default()),
        accept_length_output_tokens: 1,
        accept_length_decode_forwards: 1,
    }
}

fn publisher(
    output_tx: mpsc::UnboundedSender<Vec<OutputSignal>>,
    captured: DeferredKvPublishBuffer,
    log: Arc<Mutex<Vec<PublishedEffect>>>,
) -> LiveEffectsPublisher {
    let (admission_tx, _admission_rx) = mpsc::unbounded_channel();
    let (lifecycle_tx, _lifecycle_rx) = mpsc::channel(4);
    let (metrics_tx, _metrics_rx) = watch::channel(MockerMetrics::default());
    LiveEffectsPublisher::new(
        Some(output_tx.into()),
        Some(admission_tx),
        lifecycle_tx,
        metrics_tx,
        KvEventPublishers::default(),
        FpmPublisher::default(),
        captured,
    )
    .with_publication_log(log)
}

async fn cancel_pending_pass(
    command_result: SchedulerCommandResult,
    discard_pending_output: bool,
) -> (SchedulerCommandResult, PendingLivePass) {
    let request_id = Uuid::from_u128(1);
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(false, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        live_pass_limit: None,
        refill_command_tx: None,
        applied_command_count: 0,
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log);
    let mut pending = publisher.capture_pass(pass());
    let (_command_tx, mut command_rx) = mpsc::channel(1);
    let (cancellation_tx, mut cancellation_rx) = mpsc::channel(1);
    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    cancellation_tx
        .send(SchedulerCancellationEnvelope {
            request_id,
            discard_pending_output,
            reply,
        })
        .await
        .unwrap();

    let result = {
        let cancel_token = CancellationToken::new();
        let scheduler_start = Instant::now();
        let mut deferred_commands = VecDeque::new();
        let boundary = wait_for_live_pass_boundary(
            &mut core,
            &mut command_rx,
            &mut cancellation_rx,
            &mut deferred_commands,
            &mut pending,
            &publisher,
            &scheduler_start,
            &cancel_token,
            Instant::now() + Duration::from_secs(5),
        );
        tokio::pin!(boundary);
        let result = tokio::select! {
            reply = reply_rx => reply.unwrap().unwrap().result,
            boundary_result = &mut boundary => {
                panic!("pass boundary completed before cancellation was acknowledged: {boundary_result}")
            }
        };
        cancel_token.cancel();
        assert!(!boundary.await);
        result
    };
    (result, pending)
}

#[tokio::test]
async fn midpass_cancellation_is_observed_without_controls_and_suppresses_pending_output() {
    let (result, pending) = cancel_pending_pass(SchedulerCommandResult::Applied, false).await;

    assert_eq!(result, SchedulerCommandResult::Applied);
    assert!(pending.pass.output_signals.is_empty());
    assert_eq!(pending.pass.completed_requests, 0);
    assert_eq!(pending.pass.accept_length_output_tokens, 0);
}

#[tokio::test]
async fn explicit_discard_suppresses_pending_output_after_noop_cancellation() {
    let (result, pending) = cancel_pending_pass(SchedulerCommandResult::Noop, true).await;

    assert_eq!(result, SchedulerCommandResult::Noop);
    assert!(pending.pass.output_signals.is_empty());
    assert_eq!(pending.pass.completed_requests, 0);
    assert_eq!(pending.pass.accept_length_output_tokens, 0);
}

#[tokio::test]
async fn pass_effects_publish_once_in_boundary_order_and_isolate_midpass_ack() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result: SchedulerCommandResult::Applied,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        live_pass_limit: None,
        refill_command_tx: None,
        applied_command_count: 0,
    };
    core.publish_kv(1);
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    drop(output_rx);
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let mut pending = publisher.capture_pass(pass());
    publisher.publish_pass_start(&mut pending);

    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelSource {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[PublishedEffect::Admissions, PublishedEffect::Ack]
    );

    publisher.publish_pass(&mut core, pending).await;
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Ack,
            PublishedEffect::Kv,
            PublishedEffect::Fpm,
            PublishedEffect::Outputs,
            PublishedEffect::Accounting,
            PublishedEffect::Kv,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn controlled_pass_start_router_effects_precede_midpass_ack_without_duplicates() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result: SchedulerCommandResult::Applied,
        command_effects: false,
        midpass_kv_effects: true,
        execute_count: 0,
        live_pass_limit: None,
        refill_command_tx: None,
        applied_command_count: 0,
    };
    core.publish_kv(1);
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let mut pass = pass();
    pass.router_event_visibility = RouterEventVisibility::PassStart;
    let mut pending = publisher.capture_pass(pass);

    publisher.publish_pass_start(&mut pending);
    let (reply, reply_rx) = tokio::sync::oneshot::channel();
    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::CancelSource {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            false,
            1.0,
        )
        .await;
    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Kv,
            PublishedEffect::Ack,
        ]
    );

    publisher.publish_pass(&mut core, pending).await;
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Admissions,
            PublishedEffect::Kv,
            PublishedEffect::Ack,
            PublishedEffect::Kv,
            PublishedEffect::Fpm,
            PublishedEffect::Outputs,
            PublishedEffect::Accounting,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

#[tokio::test]
async fn command_effects_publish_kv_before_ack_then_lifecycle_and_metrics() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result: SchedulerCommandResult::Applied,
        command_effects: true,
        midpass_kv_effects: false,
        execute_count: 0,
        live_pass_limit: None,
        refill_command_tx: None,
        applied_command_count: 0,
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let publisher = publisher(output_tx, captured, log.clone());
    let (reply, reply_rx) = tokio::sync::oneshot::channel();

    publisher
        .apply_command(
            &mut core,
            SchedulerCommandEnvelope {
                command: SchedulerCommand::ActivateDestination {
                    handoff_id: HandoffId::from(Uuid::from_u128(2)),
                },
                reply,
            },
            true,
            1.0,
        )
        .await;

    assert_eq!(
        reply_rx.await.unwrap().unwrap().result,
        SchedulerCommandResult::Applied
    );
    assert_eq!(
        log.lock().unwrap().as_slice(),
        &[
            PublishedEffect::Kv,
            PublishedEffect::Ack,
            PublishedEffect::Lifecycle,
            PublishedEffect::Metrics,
        ]
    );
}

// Regression: a productive zero-duration core could monopolize a current-thread
// runtime, preventing an external task from triggering shutdown.
#[tokio::test(flavor = "current_thread")]
async fn external_shutdown_stops_a_nonempty_zero_duration_progress_loop() {
    let cancel_token = CancellationToken::new();
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(false, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result: SchedulerCommandResult::Applied,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        live_pass_limit: Some(10_000),
        refill_command_tx: None,
        applied_command_count: 0,
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let publisher = publisher(output_tx, captured, Arc::new(Mutex::new(Vec::new())));
    let (_request_tx, request_rx) = mpsc::unbounded_channel();
    let (_command_tx, command_rx) = mpsc::channel(1);
    let (_cancellation_tx, cancellation_rx) = mpsc::channel(1);
    let external_cancel_token = cancel_token.clone();
    let cancel_task = tokio::spawn(async move {
        external_cancel_token.cancel();
    });

    run_live_scheduler(
        &mut core,
        request_rx,
        command_rx,
        cancellation_rx,
        publisher,
        cancel_token,
    )
    .await;

    cancel_task.await.unwrap();
    assert!(core.execute_count > 0);
}

#[tokio::test]
async fn pre_pass_control_drain_is_bounded_to_the_entry_snapshot() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(false, false);
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let publisher = publisher(output_tx, captured, Arc::new(Mutex::new(Vec::new())));
    let (_request_tx, mut request_rx) = mpsc::unbounded_channel();
    let (command_tx, mut command_rx) = mpsc::channel(1);
    let (_cancellation_tx, mut cancellation_rx) = mpsc::channel(1);
    let (reply, _reply_rx) = tokio::sync::oneshot::channel();
    command_tx
        .try_send(SchedulerCommandEnvelope {
            command: SchedulerCommand::CancelSource {
                handoff_id: HandoffId::from(Uuid::from_u128(2)),
            },
            reply,
        })
        .unwrap();
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_result: SchedulerCommandResult::Applied,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        live_pass_limit: Some(1),
        refill_command_tx: Some(command_tx),
        applied_command_count: 0,
    };

    assert!(
        receive_until_live_schedulable(
            &mut core,
            &mut request_rx,
            &mut command_rx,
            &mut cancellation_rx,
            &publisher,
            &Instant::now(),
            &CancellationToken::new(),
        )
        .await
    );
    assert_eq!(core.applied_command_count, 1);
    assert_eq!(command_rx.len(), 1);
}
