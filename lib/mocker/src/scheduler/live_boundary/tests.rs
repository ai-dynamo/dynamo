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
    command_effects: bool,
    midpass_kv_effects: bool,
    execute_count: usize,
    cancel_after_execute: Option<(usize, CancellationToken)>,
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
        self.cancel_after_execute.is_none()
    }

    fn receive_live_request(&mut self, _request: DirectRequest) {}

    fn execute_live_pass(&mut self, _scheduler_start: &Instant) -> LivePassExecution {
        let (cancel_after, cancel_token) = self
            .cancel_after_execute
            .as_ref()
            .expect("boundary-only fake does not execute live passes");
        self.execute_count += 1;
        assert!(
            self.execute_count <= *cancel_after,
            "scheduler did not observe shutdown between productive zero-duration passes"
        );
        if self.execute_count == *cancel_after {
            cancel_token.cancel();
        }
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
        let mut effects = SchedulerCommandEffects::new(SchedulerCommandResult::Applied);
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
            token_id: None,
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
        Some(output_tx),
        Some(admission_tx),
        lifecycle_tx,
        metrics_tx,
        KvEventPublishers::default(),
        FpmPublisher::default(),
        captured,
    )
    .with_publication_log(log)
}

#[tokio::test]
async fn pass_effects_publish_once_in_boundary_order_and_isolate_midpass_ack() {
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(true, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        cancel_after_execute: None,
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
        command_effects: false,
        midpass_kv_effects: true,
        execute_count: 0,
        cancel_after_execute: None,
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
        command_effects: true,
        midpass_kv_effects: false,
        execute_count: 0,
        cancel_after_execute: None,
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

// Regression: a productive zero-duration core could spin forever without
// observing shutdown because it never entered a cancellation-aware wait.
#[tokio::test]
async fn shutdown_stops_a_nonempty_zero_duration_progress_loop() {
    let cancel_token = CancellationToken::new();
    let (captured, buffering_publishers) = capture_deferred_kv_publish_sink(false, false);
    let mut core = FakeCore {
        publishers: buffering_publishers,
        command_effects: false,
        midpass_kv_effects: false,
        execute_count: 0,
        cancel_after_execute: Some((3, cancel_token.clone())),
    };
    let (output_tx, _output_rx) = mpsc::unbounded_channel();
    let publisher = publisher(output_tx, captured, Arc::new(Mutex::new(Vec::new())));
    let (_request_tx, request_rx) = mpsc::unbounded_channel();
    let (_command_tx, command_rx) = mpsc::channel(1);

    run_live_scheduler(
        &mut core,
        request_rx,
        command_rx,
        publisher,
        cancel_token,
        false,
    )
    .await;

    assert_eq!(core.execute_count, 3);
}
