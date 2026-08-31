// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use async_trait::async_trait;
use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData};
use dynamo_mocker::common::handoff::{
    HandoffTransferTiming, NormalizedHandoffConformance, NormalizedHandoffEvent,
    NormalizedStoredTiming,
};
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

use std::collections::HashSet;

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

struct SemanticEvent {
    event: LiveHandoffEvent,
    consumed: oneshot::Sender<()>,
}

struct SemanticEventSender {
    events: mpsc::UnboundedSender<SemanticEvent>,
}

impl SemanticEventSender {
    fn send(&self, event: LiveHandoffEvent) -> oneshot::Receiver<()> {
        let (consumed, consumed_rx) = oneshot::channel();
        assert!(
            self.events.send(SemanticEvent { event, consumed }).is_ok(),
            "semantic handoff event stream closed"
        );
        consumed_rx
    }
}

struct SemanticEvents {
    events: mpsc::UnboundedReceiver<SemanticEvent>,
}

#[async_trait]
impl HandoffEventStream for SemanticEvents {
    async fn recv(&mut self) -> Option<LiveHandoffEvent> {
        let event = self.events.recv().await?;
        let _ = event.consumed.send(());
        Some(event.event)
    }
}

fn semantic_boundary() -> (
    HandoffControl,
    mpsc::UnboundedReceiver<ControlInvocation>,
    HandoffEvents,
    SemanticEventSender,
) {
    let (call_tx, call_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    (
        HandoffControl::new(Arc::new(SemanticControl { calls: call_tx })),
        call_rx,
        HandoffEvents::new(Box::new(SemanticEvents { events: event_rx })),
        SemanticEventSender { events: event_tx },
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

    let consumed = event_tx.send(LiveHandoffEvent::DestinationReserved {
        transferable_prompt_tokens: 4,
    });
    consumed
        .await
        .expect("destination session should consume the reservation event");
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
async fn destination_rejects_an_action_for_another_handoff() {
    let request_id = Uuid::from_u128(70_100);
    let handoff_id = HandoffId::from(Uuid::from_u128(70_101));
    let other_handoff_id = HandoffId::from(Uuid::from_u128(70_102));
    let (server, mut source, destination, shutdown) = bootstrap_pair(
        handoff_id,
        request_id,
        HandoffOrder::DestinationFirst,
        EngineType::Sglang,
    )
    .await;
    let (control, mut calls, events, _event_tx) = semantic_boundary();
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
    let mut other_coordinator =
        HandoffCoordinatorCore::new(other_handoff_id, HandoffOrder::DestinationFirst);
    let reserve = other_coordinator.start().unwrap().pop().unwrap();
    source
        .send(BootstrapMessage::Action(reserve))
        .await
        .unwrap();
    let response = tokio::select! {
        response = source.recv() => response.unwrap(),
        invocation = calls.recv() => {
            panic!(
                "mismatched action reached typed control: {:?}",
                invocation.unwrap().action
            );
        }
    };
    assert!(matches!(
        response,
        Some(BootstrapMessage::ActionAck {
            action_id,
            outcome: HandoffActionOutcome::Failed(message),
        }) if action_id == reserve.id && message.contains("does not match bootstrap handoff")
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
    let _consumed = event_tx.send(LiveHandoffEvent::DestinationReserved {
        transferable_prompt_tokens: 4,
    });
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
    let consumed = event_tx.send(LiveHandoffEvent::SourceHeld {
        transfer_timing: transfer_timing(None),
    });
    consumed
        .await
        .expect("source session should consume the held event");
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

#[tokio::test]
async fn destination_cleanup_abandons_an_unsubmitted_registration() {
    let engine = LiveEngine::start(
        args_with_mode(
            EngineType::Vllm,
            WorkerType::Decode,
            KvTransferTimingMode::FullPrompt,
        ),
        0,
    )
    .unwrap();
    let handoff_id = HandoffId::from(Uuid::from_u128(74_001));
    let request_id = Uuid::from_u128(74_002);
    let (registration, request_stream) = engine.prepare_request(request(request_id, 1)).unwrap();
    let (control, events) = engine.register_handoff(handoff_id).unwrap();
    let (control, events) = live_handoff_boundary(control, events, registration);
    let retained_control = control.clone();

    assert_eq!(
        control
            .execute(HandoffControlAction::CancelDestination)
            .await
            .unwrap(),
        HandoffActionOutcome::Applied
    );
    assert!(
        !tokio::time::timeout(Duration::from_secs(1), request_stream.cancel())
            .await
            .expect("prepared request cancellation should not hang")
            .unwrap()
    );
    wait_for_idle(&engine).await;
    assert_eq!(engine.metrics_receiver().borrow().active_decode_blocks, 0);

    let (replacement, replacement_stream) = engine
        .prepare_request(request(request_id, 1))
        .expect("abandoned client request ID should be reusable");
    drop(replacement);
    drop(replacement_stream);
    drop(control);
    drop(retained_control);
    drop(events);
    let (replacement_control, replacement_events) = engine
        .register_handoff(handoff_id)
        .expect("abandoned handoff ID should be reusable");
    drop(replacement_control);
    drop(replacement_events);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn active_handoff_shutdown_releases_scheduler_and_session_ownership() {
    let engine_type = EngineType::Vllm;
    let transfer_timing_mode = KvTransferTimingMode::FullPrompt;
    let (source_engine, _source_kv) =
        start_live_engine(engine_type, WorkerType::Prefill, transfer_timing_mode);
    let (destination_engine, _destination_kv) =
        start_live_engine(engine_type, WorkerType::Decode, transfer_timing_mode);
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let manager = SourceHandoffManager::start(
        server.take_incoming_receiver().unwrap(),
        1,
        Duration::from_secs(2),
        shutdown.clone(),
    );
    let handoff_id = HandoffId::from(Uuid::from_u128(75_001));
    let request_id = Uuid::from_u128(75_002);
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 20,
        request_id,
    };
    let order = order_for_engine(engine_type).unwrap();

    let (source_registration, source_request) = source_engine
        .prepare_request(request(request_id, 1))
        .unwrap();
    let (source_control, source_events) = source_engine.register_handoff(handoff_id).unwrap();
    let (source_control, source_events) =
        live_handoff_boundary(source_control, source_events, source_registration);
    let (destination_registration, destination_request) = destination_engine
        .prepare_request(request(request_id, 2))
        .unwrap();
    let (destination_control, destination_events) =
        destination_engine.register_handoff(handoff_id).unwrap();
    let (destination_control, destination_events) = live_handoff_boundary(
        destination_control,
        destination_events,
        destination_registration,
    );

    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let permit = permits.clone().try_acquire_owned().unwrap();
    let (completion_tx, completion_rx) = oneshot::channel();
    let (observer_tx, mut observer_rx) = mpsc::unbounded_channel();
    manager
        .try_register(SourceRegistration {
            identity: identity.clone(),
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
    let destination_connection = connect_to_prefill(
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
    let destination_session = tokio::spawn(run_destination_session(
        destination_connection,
        destination_control,
        destination_events,
        CancellationToken::new(),
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    tokio::time::timeout(Duration::from_secs(2), async {
        let mut source_held = false;
        let mut destination_reserved = false;
        while !source_held || !destination_reserved {
            match observer_rx.recv().await.unwrap() {
                NormalizedHandoffEvent::SourceHeld => source_held = true,
                NormalizedHandoffEvent::DestinationReserved => destination_reserved = true,
                _ => {}
            }
        }
    })
    .await
    .expect("handoff should acquire source and destination ownership");

    shutdown.cancel();
    let (source_completion, destination_completion) =
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(completion_rx, destination_session)
        })
        .await
        .expect("active handoff shutdown should join both participants");
    assert!(source_completion.unwrap().is_err());
    assert!(destination_completion.unwrap().is_err());
    manager.wait_closed().await;
    server.wait_closed().await;

    let (source_cleanup, destination_cleanup) =
        tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(source_request.cancel(), destination_request.cancel())
        })
        .await
        .expect("request cleanup should complete after handoff shutdown");
    source_cleanup.unwrap();
    destination_cleanup.unwrap();
    wait_for_idle(&source_engine).await;
    wait_for_idle(&destination_engine).await;
    assert_eq!(
        destination_engine
            .metrics_receiver()
            .borrow()
            .active_decode_blocks,
        0
    );
    assert_eq!(permits.available_permits(), 1);

    let (source_replacement, source_replacement_events) =
        source_engine.register_handoff(handoff_id).unwrap();
    drop(source_replacement);
    drop(source_replacement_events);
    let (destination_replacement, destination_replacement_events) =
        destination_engine.register_handoff(handoff_id).unwrap();
    drop(destination_replacement);
    drop(destination_replacement_events);
    assert!(probe_engine_drained(&source_engine).await);
    assert!(probe_engine_drained(&destination_engine).await);
    source_engine.shutdown().await.unwrap();
    destination_engine.shutdown().await.unwrap();
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

fn drain_stored_hashes(
    events: &mut mpsc::UnboundedReceiver<KvCacheEvent>,
) -> Vec<ExternalSequenceBlockHash> {
    std::iter::from_fn(|| events.try_recv().ok())
        .flat_map(|event| match event.data {
            KvCacheEventData::Stored(data) => data
                .blocks
                .into_iter()
                .map(|block| block.block_hash)
                .collect(),
            KvCacheEventData::Removed(_) | KvCacheEventData::Cleared => Vec::new(),
        })
        .collect()
}

struct LiveHandoffObservation {
    lifecycle: Vec<NormalizedHandoffEvent>,
    before_activation: usize,
    activation_hashes: Vec<ExternalSequenceBlockHash>,
    remaining_kv: mpsc::UnboundedReceiver<KvCacheEvent>,
}

async fn observe_live_handoff(
    mut lifecycle_rx: mpsc::UnboundedReceiver<NormalizedHandoffEvent>,
    mut destination_kv: mpsc::UnboundedReceiver<KvCacheEvent>,
) -> LiveHandoffObservation {
    let mut lifecycle = Vec::new();
    let mut before_activation = 0;
    let mut activation_hashes = Vec::new();
    let mut activated = false;
    loop {
        let event = lifecycle_rx
            .recv()
            .await
            .expect("handoff observer closed before completion");
        lifecycle.push(event);
        match event {
            NormalizedHandoffEvent::DestinationActivated => {
                activation_hashes.extend(drain_stored_hashes(&mut destination_kv));
                activated = true;
            }
            NormalizedHandoffEvent::Completed => {
                return LiveHandoffObservation {
                    lifecycle,
                    before_activation,
                    activation_hashes,
                    remaining_kv: destination_kv,
                };
            }
            _ if !activated => {
                before_activation += drain_stored_hashes(&mut destination_kv).len();
            }
            _ => {}
        }
    }
}

async fn wait_for_idle(engine: &LiveEngine) {
    let result = tokio::time::timeout(Duration::from_secs(2), async {
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
    .await;
    if result.is_err() {
        let metrics = engine.metrics_receiver().borrow().clone();
        panic!(
            "live handoff engine must return to idle: routes={}, running={}, waiting={}, active_blocks={}",
            engine.active_request_count(),
            metrics.running_requests,
            metrics.waiting_requests,
            metrics.active_decode_blocks,
        );
    }
}

async fn probe_engine_drained(engine: &LiveEngine) -> bool {
    let mut probe = engine
        .submit(dynamo_mocker::common::protocols::DirectRequest {
            tokens: (10_000..10_252).collect(),
            max_output_tokens: 1,
            output_token_ids: Some(vec![42]),
            uuid: Some(Uuid::new_v4()),
            ..Default::default()
        })
        .await
        .expect("drain probe submission failed");
    let completed = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(signal) = probe.recv().await {
            if signal.completed {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    drop(probe);
    wait_for_idle(engine).await;
    completed && engine.active_request_count() == 0
}

#[tokio::test]
async fn live_and_offline_handoff_surfaces_share_one_conformance_report() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        for transfer_timing_mode in [
            KvTransferTimingMode::FullPrompt,
            KvTransferTimingMode::DestinationMissing,
        ] {
            for source_arrives_first in [true, false] {
                let (source_engine, mut source_kv) =
                    start_live_engine(engine_type, WorkerType::Prefill, transfer_timing_mode);
                let (destination_engine, destination_kv) =
                    start_live_engine(engine_type, WorkerType::Decode, transfer_timing_mode);
                let shutdown = CancellationToken::new();
                let server =
                    BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
                        .await
                        .unwrap();
                let incoming = server.take_incoming_receiver().unwrap();
                let manager = SourceHandoffManager::start(
                    incoming,
                    1,
                    Duration::from_secs(2),
                    shutdown.clone(),
                );
                let handoff_id = HandoffId::new();
                let request_id = Uuid::new_v4();
                let identity = BootstrapIdentity {
                    handoff_id,
                    bootstrap_room: 19,
                    request_id,
                };
                let order = order_for_engine(engine_type).unwrap();

                let (source_registration, source_request) = source_engine
                    .prepare_request(request(request_id, 1))
                    .unwrap();
                let (source_control, source_events) =
                    source_engine.register_handoff(handoff_id).unwrap();
                let (source_control, source_events) =
                    live_handoff_boundary(source_control, source_events, source_registration);
                let (destination_registration, destination_request) = destination_engine
                    .prepare_request(request(request_id, 2))
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
                let (observer_tx, observer_rx) = mpsc::unbounded_channel();
                let observer = tokio::spawn(observe_live_handoff(observer_rx, destination_kv));
                let mut source = Some(SourceRegistration {
                    identity: identity.clone(),
                    order,
                    engine_type,
                    control: source_control,
                    lifecycle: source_events,
                    completion_tx,
                    cancel: CancellationToken::new(),
                    observer: Some(observer_tx),
                    _permit: permit,
                });
                if source_arrives_first {
                    manager.try_register(source.take().unwrap()).unwrap();
                }
                let destination_connection = connect_to_prefill(
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
                if !source_arrives_first {
                    manager.wait_for_pending_destination(handoff_id).await;
                    manager.try_register(source.take().unwrap()).unwrap();
                }
                let destination_session = tokio::spawn(run_destination_session(
                    destination_connection,
                    destination_control,
                    destination_events,
                    CancellationToken::new(),
                    Duration::from_secs(2),
                    shutdown.clone(),
                ));

                let (
                    source_output,
                    destination_output,
                    source_completion,
                    destination_completion,
                    observation,
                ) = tokio::time::timeout(Duration::from_secs(5), async {
                    tokio::join!(
                        collect_output(source_request),
                        collect_output(destination_request),
                        completion_rx,
                        destination_session,
                        observer,
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

                let mut observation = observation.unwrap();
                let activation_set = observation
                    .activation_hashes
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                let repeated_activation_hashes_after_activation =
                    drain_stored_hashes(&mut observation.remaining_kv)
                        .into_iter()
                        .filter(|hash| activation_set.contains(hash))
                        .count();
                assert!(!drain_stored_hashes(&mut source_kv).is_empty());
                wait_for_idle(&source_engine).await;
                wait_for_idle(&destination_engine).await;
                let source_drained = probe_engine_drained(&source_engine).await;
                let destination_drained = probe_engine_drained(&destination_engine).await;
                let source_route_reusable = source_engine.register_handoff(handoff_id).is_ok();
                let destination_route_reusable =
                    destination_engine.register_handoff(handoff_id).is_ok();
                let report = NormalizedHandoffConformance {
                    engine_type,
                    order,
                    lifecycle: observation.lifecycle,
                    source_output_tokens: source_output
                        .iter()
                        .filter(|signal| signal.token_id.is_some())
                        .count(),
                    destination_output_tokens: destination_output
                        .iter()
                        .filter(|signal| signal.token_id.is_some())
                        .count(),
                    completed_requests: destination_output
                        .iter()
                        .filter(|signal| signal.completed)
                        .count(),
                    destination_stored: NormalizedStoredTiming {
                        before_activation: observation.before_activation,
                        on_activation: observation.activation_hashes.len(),
                        repeated_activation_hashes_after_activation,
                    },
                    source_drained,
                    destination_drained,
                    driver_drained: permits.available_permits() == 1
                        && source_route_reusable
                        && destination_route_reusable,
                };
                report.validate().unwrap();
                let offline = dynamo_mocker::replay::run_offline_handoff_conformance(
                    engine_type,
                    transfer_timing_mode,
                )
                .unwrap();
                assert_eq!(report, offline);

                shutdown.cancel();
                manager.wait_closed().await;
                server.wait_closed().await;
                source_engine.shutdown().await.unwrap();
                destination_engine.shutdown().await.unwrap();
            }
        }
    }
}
