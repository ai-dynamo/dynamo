// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use async_trait::async_trait;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use dynamo_mocker::common::handoff::{HandoffTransferTiming, NormalizedHandoffEvent};
use dynamo_mocker::common::protocols::{
    EngineType, FpmPublisher, KvCacheEventSink, KvEventPublishers, KvTransferTimingMode,
    MockEngineArgs, WorkerType,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig, LiveRequest};
use dynamo_mocker::services::bootstrap::{
    BootstrapParticipantRole, BootstrapServer, BootstrapServerConfig, ParticipantRegistration,
    connect_to_prefill,
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use uuid::Uuid;

fn args_with_mode(
    engine_type: EngineType,
    worker_type: WorkerType,
    transfer_timing_mode: KvTransferTimingMode,
) -> MockEngineArgs {
    let mut builder = MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(4)
        .num_gpu_blocks(64)
        .max_num_batched_tokens(Some(64))
        .max_num_seqs(Some(2))
        .worker_type(worker_type)
        .speedup_ratio(1000.0)
        .decode_speedup_ratio(1000.0)
        .kv_transfer_bandwidth(Some(1.0))
        .kv_bytes_per_token(Some(1_000_000))
        .kv_transfer_timing_mode(transfer_timing_mode);
    if engine_type == EngineType::Sglang {
        builder = builder.sglang(Some(Default::default()));
    }
    builder.build().unwrap()
}

fn request(uuid: Uuid, output_tokens: usize) -> dynamo_mocker::common::protocols::DirectRequest {
    dynamo_mocker::common::protocols::DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: output_tokens,
        output_token_ids: Some(vec![42; output_tokens]),
        uuid: Some(uuid),
        ..Default::default()
    }
}

fn transfer_timing(delay_ms: Option<f64>) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode: KvTransferTimingMode::FullPrompt,
        full_prompt_tokens: 1,
        kv_bytes_per_token: delay_ms.map(|delay_ms| (delay_ms * 1_000_000.0) as usize),
        bandwidth_gb_s: delay_ms.map(|_| 1.0),
    }
}

#[test]
fn timeout_delay_resolves_at_the_mode_specific_boundary() {
    let full = HandoffTransferTiming {
        mode: KvTransferTimingMode::FullPrompt,
        full_prompt_tokens: 8,
        kv_bytes_per_token: Some(1_000_000),
        bandwidth_gb_s: Some(1.0),
    };
    assert_eq!(transfer_timeout_delay(full, None), Some(Some(8.0)));

    let missing = HandoffTransferTiming {
        mode: KvTransferTimingMode::DestinationMissing,
        ..full
    };
    assert_eq!(transfer_timeout_delay(missing, None), None);
    assert_eq!(transfer_timeout_delay(missing, Some(4)), Some(Some(4.0)));
}

struct ControlInvocation {
    action: HandoffControlAction,
    reply: oneshot::Sender<Result<HandoffActionOutcome>>,
}

struct SemanticControl {
    calls: mpsc::UnboundedSender<ControlInvocation>,
}

#[async_trait]
impl HandoffSchedulerControl for SemanticControl {
    async fn execute(&self, action: HandoffControlAction) -> Result<HandoffActionOutcome> {
        let (reply, response) = oneshot::channel();
        self.calls
            .send(ControlInvocation { action, reply })
            .map_err(|_| anyhow!("semantic handoff control closed"))?;
        response
            .await
            .map_err(|_| anyhow!("semantic handoff control reply dropped"))?
    }
}

struct SemanticEvents {
    events: mpsc::UnboundedReceiver<LiveHandoffEvent>,
}

#[async_trait]
impl HandoffEventStream for SemanticEvents {
    async fn recv(&mut self) -> Option<LiveHandoffEvent> {
        self.events.recv().await
    }
}

fn semantic_boundary() -> (
    HandoffControl,
    mpsc::UnboundedReceiver<ControlInvocation>,
    HandoffEvents,
    mpsc::UnboundedSender<LiveHandoffEvent>,
) {
    let (call_tx, call_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    (
        HandoffControl::new(Arc::new(SemanticControl { calls: call_tx })),
        call_rx,
        HandoffEvents::new(Box::new(SemanticEvents { events: event_rx })),
        event_tx,
    )
}

fn acknowledge(invocation: ControlInvocation, outcome: HandoffActionOutcome) {
    invocation.reply.send(Ok(outcome)).unwrap();
}

async fn bootstrap_pair(
    handoff_id: HandoffId,
    request_id: Uuid,
    order: HandoffOrder,
    engine_type: EngineType,
) -> (
    Arc<BootstrapServer>,
    BootstrapConnection,
    BootstrapConnection,
    CancellationToken,
) {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 17,
        request_id,
    };
    let destination = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order,
            engine_type,
        },
    )
    .await
    .unwrap();
    let source = incoming.recv().await.unwrap().connection;
    (server, source, destination, shutdown)
}

async fn finish_test_transport(server: Arc<BootstrapServer>, shutdown: CancellationToken) {
    shutdown.cancel();
    server.wait_closed().await;
}

#[tokio::test]
async fn destination_ack_precedes_an_early_reservation_fact() {
    let request_id = Uuid::from_u128(70_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(70_001));
    let (server, mut source, destination, shutdown) = bootstrap_pair(
        handoff_id,
        request_id,
        HandoffOrder::DestinationFirst,
        EngineType::Sglang,
    )
    .await;
    let (control, mut calls, events, event_tx) = semantic_boundary();
    let cancel = CancellationToken::new();
    let session = tokio::spawn(run_destination_session(
        destination,
        control,
        events,
        cancel.clone(),
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    source.send(BootstrapMessage::Registered).await.unwrap();
    let mut coordinator = HandoffCoordinatorCore::new(handoff_id, HandoffOrder::DestinationFirst);
    let reserve = coordinator.start().unwrap().pop().unwrap();
    source
        .send(BootstrapMessage::Action(reserve))
        .await
        .unwrap();
    let invocation = calls.recv().await.unwrap();
    assert_eq!(invocation.action, HandoffControlAction::ReserveDestination);

    event_tx
        .send(LiveHandoffEvent::DestinationReserved {
            transferable_prompt_tokens: 4,
        })
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), source.recv())
            .await
            .is_err(),
        "reservation fact must wait for the scheduler acknowledgement"
    );

    acknowledge(invocation, HandoffActionOutcome::Accepted);
    assert!(matches!(
        source.recv().await.unwrap(),
        Some(BootstrapMessage::ActionAck {
            action_id,
            outcome: HandoffActionOutcome::Accepted,
        }) if action_id == reserve.id
    ));
    assert!(matches!(
        source.recv().await.unwrap(),
        Some(BootstrapMessage::Fact(HandoffFact::DestinationReserved {
            handoff_id: observed,
            transferable_prompt_tokens: 4,
        })) if observed == handoff_id
    ));

    cancel.cancel();
    let cleanup = calls.recv().await.unwrap();
    assert_eq!(cleanup.action, HandoffControlAction::CancelDestination);
    acknowledge(cleanup, HandoffActionOutcome::Applied);
    assert!(session.await.unwrap().is_err());
    finish_test_transport(server, shutdown).await;
}

#[tokio::test]
async fn premature_complete_waits_for_destination_cleanup() {
    let request_id = Uuid::from_u128(71_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(71_001));
    let (server, mut source, destination, shutdown) = bootstrap_pair(
        handoff_id,
        request_id,
        HandoffOrder::DestinationFirst,
        EngineType::Sglang,
    )
    .await;
    let (control, mut calls, events, event_tx) = semantic_boundary();
    let session = tokio::spawn(run_destination_session(
        destination,
        control,
        events,
        CancellationToken::new(),
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    source.send(BootstrapMessage::Registered).await.unwrap();
    let mut coordinator = HandoffCoordinatorCore::new(handoff_id, HandoffOrder::DestinationFirst);
    let reserve = coordinator.start().unwrap().pop().unwrap();
    source
        .send(BootstrapMessage::Action(reserve))
        .await
        .unwrap();
    acknowledge(calls.recv().await.unwrap(), HandoffActionOutcome::Accepted);
    event_tx
        .send(LiveHandoffEvent::DestinationReserved {
            transferable_prompt_tokens: 4,
        })
        .unwrap();
    let _ = source.recv().await.unwrap();
    let _ = source.recv().await.unwrap();

    source.send(BootstrapMessage::Complete).await.unwrap();
    let cleanup = calls.recv().await.unwrap();
    assert_eq!(cleanup.action, HandoffControlAction::CancelDestination);
    let mut session = Box::pin(session);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut session)
            .await
            .is_err(),
        "destination session must retain cleanup ownership until acknowledgement"
    );
    acknowledge(cleanup, HandoffActionOutcome::Applied);
    assert!(session.await.unwrap().is_err());
    finish_test_transport(server, shutdown).await;
}

#[tokio::test]
async fn source_held_waits_for_submit_outcome_before_progressing() {
    let request_id = Uuid::from_u128(72_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(72_001));
    let (server, source_connection, mut destination, shutdown) = bootstrap_pair(
        handoff_id,
        request_id,
        HandoffOrder::SourceFirst,
        EngineType::Vllm,
    )
    .await;
    let (control, mut calls, events, event_tx) = semantic_boundary();
    let cancel = CancellationToken::new();
    let (completion_tx, completion_rx) = oneshot::channel();
    let permit = Arc::new(tokio::sync::Semaphore::new(1))
        .try_acquire_owned()
        .unwrap();
    let session = tokio::spawn(run_source_session(
        SourceRegistration {
            identity: BootstrapIdentity {
                handoff_id,
                bootstrap_room: 17,
                request_id,
            },
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            control,
            lifecycle: events,
            completion_tx,
            cancel: cancel.clone(),
            observer: None,
            _permit: permit,
        },
        source_connection,
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Registered)
    ));
    let submit = calls.recv().await.unwrap();
    assert_eq!(submit.action, HandoffControlAction::SubmitPrefill);
    event_tx
        .send(LiveHandoffEvent::SourceHeld {
            transfer_timing: transfer_timing(None),
        })
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(20), destination.recv())
            .await
            .is_err(),
        "source-held fact must wait for submit acknowledgement"
    );

    acknowledge(submit, HandoffActionOutcome::Submitted);
    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Fact(HandoffFact::SourceHeld {
            handoff_id: observed,
            ..
        })) if observed == handoff_id
    ));

    cancel.cancel();
    let cleanup = calls.recv().await.unwrap();
    assert_eq!(cleanup.action, HandoffControlAction::CancelSource);
    acknowledge(cleanup, HandoffActionOutcome::Applied);
    assert!(session.await.unwrap().is_err());
    assert!(completion_rx.await.unwrap().is_err());
    finish_test_transport(server, shutdown).await;
}

#[tokio::test]
async fn pending_source_cancellation_releases_session_permit() {
    let (_incoming_tx, incoming_rx) = mpsc::channel(1);
    let shutdown = CancellationToken::new();
    let manager = SourceHandoffManager::start_with_rendezvous_timeout(
        incoming_rx,
        1,
        Duration::from_secs(1),
        Duration::from_secs(30),
        shutdown.clone(),
    );
    let handoff_id = HandoffId::from(Uuid::from_u128(73_001));
    let request_id = Uuid::from_u128(73_002);
    let (control, _calls, events, _event_tx) = semantic_boundary();
    let cancel = CancellationToken::new();
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let permit = permits.clone().try_acquire_owned().unwrap();
    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(SourceRegistration {
            identity: BootstrapIdentity {
                handoff_id,
                bootstrap_room: 18,
                request_id,
            },
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            control,
            lifecycle: events,
            completion_tx,
            cancel: cancel.clone(),
            observer: None,
            _permit: permit,
        })
        .unwrap();
    manager.wait_for_pending_source(handoff_id).await;

    cancel.cancel();
    assert!(completion_rx.await.unwrap().is_err());
    manager.wait_for_retired(handoff_id).await;
    assert_eq!(permits.available_permits(), 1);

    shutdown.cancel();
    manager.wait_closed().await;
}

#[derive(Clone)]
struct CapturingKvSink {
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvCacheEventSink for CapturingKvSink {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.tx
            .send(event)
            .map_err(|_| anyhow!("KV event receiver closed"))
    }
}

fn start_live_engine(
    engine_type: EngineType,
    worker_type: WorkerType,
    transfer_timing_mode: KvTransferTimingMode,
) -> (LiveEngine, mpsc::UnboundedReceiver<KvCacheEvent>) {
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let engine = LiveEngine::start_with_config(
        args_with_mode(engine_type, worker_type, transfer_timing_mode),
        0,
        LiveEngineConfig {
            kv_event_publishers: KvEventPublishers::new(
                Some(Arc::new(CapturingKvSink { tx: event_tx })),
                None,
            ),
            fpm_publisher: FpmPublisher::default(),
        },
    )
    .unwrap();
    (engine, event_rx)
}

async fn collect_output(
    mut request: LiveRequest,
) -> Vec<dynamo_mocker::common::protocols::OutputSignal> {
    let mut output = Vec::new();
    while let Some(signal) = request.recv().await {
        let terminal = signal.completed;
        output.push(signal);
        if terminal {
            break;
        }
    }
    output
}

fn stored_event_count(events: &mut mpsc::UnboundedReceiver<KvCacheEvent>) -> usize {
    std::iter::from_fn(|| events.try_recv().ok())
        .map(|event| match event.data {
            KvCacheEventData::Stored(data) => data.blocks.len(),
            KvCacheEventData::Removed(_) | KvCacheEventData::Cleared => 0,
        })
        .sum()
}

async fn wait_for_idle(engine: &LiveEngine) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let metrics = engine.metrics_receiver().borrow().clone();
            if engine.active_request_count() == 0
                && metrics.running_requests == 0
                && metrics.waiting_requests == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("live handoff engine must return to idle");
}

#[tokio::test]
async fn live_handoff_preserves_timing_kv_and_cleanup_for_supported_engines() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        for transfer_timing_mode in [
            KvTransferTimingMode::FullPrompt,
            KvTransferTimingMode::DestinationMissing,
        ] {
            let (source_engine, mut source_kv) =
                start_live_engine(engine_type, WorkerType::Prefill, transfer_timing_mode);
            let (destination_engine, mut destination_kv) =
                start_live_engine(engine_type, WorkerType::Decode, transfer_timing_mode);
            let shutdown = CancellationToken::new();
            let server =
                BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
                    .await
                    .unwrap();
            let incoming = server.take_incoming_receiver().unwrap();
            let manager =
                SourceHandoffManager::start(incoming, 1, Duration::from_secs(2), shutdown.clone());
            let handoff_id = HandoffId::new();
            let request_id = Uuid::new_v4();
            let identity = BootstrapIdentity {
                handoff_id,
                bootstrap_room: 19,
                request_id,
            };
            let order = order_for_engine(engine_type).unwrap();
            let destination_connection = connect_to_prefill(
                "127.0.0.1",
                server.port(),
                identity.clone(),
                ParticipantRegistration {
                    role: BootstrapParticipantRole::Destination,
                    dp_rank: 0,
                    order,
                    engine_type,
                },
            )
            .await
            .unwrap();

            let (source_registration, source_request) = source_engine
                .prepare_request(request(request_id, 1))
                .unwrap();
            let (source_control, source_events) =
                source_engine.register_handoff(handoff_id).unwrap();
            let (source_control, source_events) =
                live_handoff_boundary(source_control, source_events, source_registration);
            let (destination_registration, destination_request) = destination_engine
                .prepare_request(request(request_id, 1))
                .unwrap();
            let (destination_control, destination_events) =
                destination_engine.register_handoff(handoff_id).unwrap();
            let (destination_control, destination_events) = live_handoff_boundary(
                destination_control,
                destination_events,
                destination_registration,
            );

            let permits = Arc::new(tokio::sync::Semaphore::new(1));
            let permit: OwnedSemaphorePermit = permits.clone().try_acquire_owned().unwrap();
            let (completion_tx, completion_rx) = oneshot::channel();
            let (observer_tx, mut observer_rx) = mpsc::unbounded_channel();
            manager
                .try_register(SourceRegistration {
                    identity,
                    order,
                    engine_type,
                    control: source_control,
                    lifecycle: source_events,
                    completion_tx,
                    cancel: CancellationToken::new(),
                    observer: Some(observer_tx),
                    _permit: permit,
                })
                .unwrap();
            let destination_session = tokio::spawn(run_destination_session(
                destination_connection,
                destination_control,
                destination_events,
                CancellationToken::new(),
                Duration::from_secs(2),
                shutdown.clone(),
            ));

            let (source_output, destination_output, source_completion, destination_completion) =
                tokio::time::timeout(Duration::from_secs(5), async {
                    tokio::join!(
                        collect_output(source_request),
                        collect_output(destination_request),
                        completion_rx,
                        destination_session,
                    )
                })
                .await
                .expect("live handoff timed out");
            assert!(source_completion.unwrap().is_ok());
            assert!(destination_completion.unwrap().is_ok());
            assert!(source_output.last().is_some_and(|signal| signal.completed));
            assert!(
                destination_output
                    .last()
                    .is_some_and(|signal| signal.completed)
            );
            assert_eq!(permits.available_permits(), 1);

            let observed = std::iter::from_fn(|| observer_rx.try_recv().ok()).collect::<Vec<_>>();
            for expected in [
                NormalizedHandoffEvent::SourceHeld,
                NormalizedHandoffEvent::DestinationAccepted,
                NormalizedHandoffEvent::DestinationReserved,
                NormalizedHandoffEvent::DestinationActivated,
                NormalizedHandoffEvent::SourceReleased,
                NormalizedHandoffEvent::Completed,
            ] {
                assert!(
                    observed.contains(&expected),
                    "missing {expected:?} for {engine_type:?}/{transfer_timing_mode:?}: {observed:?}"
                );
            }
            assert!(stored_event_count(&mut source_kv) > 0);
            assert!(stored_event_count(&mut destination_kv) > 0);
            wait_for_idle(&source_engine).await;
            wait_for_idle(&destination_engine).await;

            shutdown.cancel();
            manager.wait_closed().await;
            server.wait_closed().await;
            source_engine.shutdown().await.unwrap();
            destination_engine.shutdown().await.unwrap();
        }
    }
}
