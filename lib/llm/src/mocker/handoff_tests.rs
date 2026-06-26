// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData};
use dynamo_mocker::common::handoff::{
    HandoffCompletion, HandoffTransferTiming, NormalizedHandoffConformance, NormalizedStoredTiming,
};
use dynamo_mocker::common::protocols::{
    FpmPublisher, KvCacheEventSink, KvEventPublishers, KvTransferTimingMode, MockEngineArgs,
    OutputSignal, WorkerType,
};
use dynamo_mocker::engine::create_engine;
use dynamo_mocker::scheduler::{MockerMetrics, SchedulerHandle};
use dynamo_mocker::services::bootstrap::{
    BootstrapParticipantRole, BootstrapServer, BootstrapServerConfig, ParticipantRegistration,
    connect_to_prefill,
};
use uuid::Uuid;

fn args(engine_type: EngineType, worker_type: WorkerType) -> MockEngineArgs {
    args_with_mode(engine_type, worker_type, KvTransferTimingMode::FullPrompt)
}

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

fn request(uuid: Uuid, output_tokens: usize) -> DirectRequest {
    DirectRequest {
        tokens: (0..8).collect(),
        max_output_tokens: output_tokens,
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

fn start_scheduler(
    engine_type: EngineType,
    worker_type: WorkerType,
    cancel: CancellationToken,
) -> (
    Box<dyn SchedulerHandle>,
    mpsc::UnboundedReceiver<Vec<OutputSignal>>,
) {
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let scheduler = create_engine(
        args(engine_type, worker_type),
        0,
        Some(output_tx),
        KvEventPublishers::default(),
        Some(cancel),
        FpmPublisher::default(),
    );
    (scheduler, output_rx)
}

fn start_scheduler_with_mode(
    engine_type: EngineType,
    worker_type: WorkerType,
    transfer_timing_mode: KvTransferTimingMode,
    cancel: CancellationToken,
) -> (
    Box<dyn SchedulerHandle>,
    mpsc::UnboundedReceiver<Vec<OutputSignal>>,
) {
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let scheduler = create_engine(
        args_with_mode(engine_type, worker_type, transfer_timing_mode),
        0,
        Some(output_tx),
        KvEventPublishers::default(),
        Some(cancel),
        FpmPublisher::default(),
    );
    (scheduler, output_rx)
}

#[derive(Clone)]
struct CapturingKvSink {
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvCacheEventSink for CapturingKvSink {
    fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
        self.tx
            .send(event)
            .map_err(|_| anyhow::anyhow!("conformance KV event receiver closed"))
    }
}

fn start_scheduler_with_kv_events_and_mode(
    engine_type: EngineType,
    worker_type: WorkerType,
    transfer_timing_mode: KvTransferTimingMode,
    cancel: CancellationToken,
) -> (
    Box<dyn SchedulerHandle>,
    mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let scheduler = create_engine(
        args_with_mode(engine_type, worker_type, transfer_timing_mode),
        0,
        Some(output_tx),
        KvEventPublishers::new(Some(Arc::new(CapturingKvSink { tx: event_tx })), None),
        Some(cancel),
        FpmPublisher::default(),
    );
    (scheduler, output_rx, event_rx)
}

fn drain_stored_hashes(
    receiver: &mut mpsc::UnboundedReceiver<KvCacheEvent>,
) -> Vec<dynamo_kv_router::protocols::ExternalSequenceBlockHash> {
    std::iter::from_fn(|| receiver.try_recv().ok())
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

async fn wait_for_scheduler_idle(mut metrics: tokio::sync::watch::Receiver<MockerMetrics>) {
    tokio::time::timeout(Duration::from_secs(2), async move {
        loop {
            let current = metrics.borrow().clone();
            if current.running_requests == 0 && current.waiting_requests == 0 {
                return;
            }
            metrics.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
}

fn spawn_lifecycle_dispatcher(
    mut receiver: mpsc::Receiver<SchedulerLifecycleEvent>,
    registry: HandoffEventRegistry,
) {
    tokio::spawn(async move {
        while let Some(event) = receiver.recv().await {
            registry.deliver(event).await;
        }
    });
}

#[derive(Default)]
struct OutputObservation {
    output_tokens: usize,
    completed_requests: usize,
}

fn observe_output_batch(
    observation: &mut OutputObservation,
    batch: Vec<OutputSignal>,
    request_id: Uuid,
) {
    for signal in batch {
        if signal.uuid != request_id || signal.rejected {
            continue;
        }
        observation.output_tokens += 1;
        observation.completed_requests += usize::from(signal.completed);
    }
}

async fn observe_through_first_terminal(
    receiver: &mut mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    request_id: Uuid,
) -> OutputObservation {
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut observation = OutputObservation::default();
        while observation.completed_requests == 0 {
            observe_output_batch(&mut observation, receiver.recv().await.unwrap(), request_id);
        }
        observation
    })
    .await
    .unwrap()
}

fn drain_output_observation(
    receiver: &mut mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    request_id: Uuid,
    observation: &mut OutputObservation,
) {
    while let Ok(batch) = receiver.try_recv() {
        observe_output_batch(observation, batch, request_id);
    }
}

#[tokio::test]
async fn destination_reservation_ack_precedes_an_early_lifecycle_fact() {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let request_id = Uuid::from_u128(70_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(70_001));
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
            order: HandoffOrder::DestinationFirst,
            engine_type: EngineType::Sglang,
        },
    )
    .await
    .unwrap();
    let mut source = incoming.recv().await.unwrap().connection;
    let registry = HandoffEventRegistry::default();
    let route = registry.register(handoff_id).unwrap();
    let (command_tx, mut command_rx) = mpsc::channel(1);
    let session = tokio::spawn(run_destination_session(
        destination,
        request(request_id, 2),
        command_tx,
        route,
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
    let envelope = command_rx.recv().await.unwrap();
    assert!(matches!(
        envelope.command,
        SchedulerCommand::ReserveDestination {
            handoff_id: observed,
            ..
        } if observed == handoff_id
    ));

    registry
        .deliver_and_wait(SchedulerLifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens: 1,
        })
        .await;
    envelope
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::DestinationAccepted { request_id },
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();

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
            ..
        })) if observed == handoff_id
    ));

    assert!(
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap()
            .is_empty()
    );
    let submit = coordinator
        .on_fact(HandoffFact::DestinationReserved {
            handoff_id,
            transferable_prompt_tokens: 1,
        })
        .unwrap()
        .pop()
        .unwrap();
    assert!(matches!(submit.action, HandoffAction::SubmitPrefill { .. }));
    assert!(
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap()
            .is_empty()
    );
    let transfer = coordinator
        .on_fact(HandoffFact::SourceHeld {
            handoff_id,
            transfer_timing: transfer_timing(Some(0.0)),
        })
        .unwrap()
        .pop()
        .unwrap();
    assert!(matches!(
        transfer.action,
        HandoffAction::StartTransfer { .. }
    ));
    assert!(
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap()
            .is_empty()
    );
    let activate = coordinator
        .on_fact(HandoffFact::TransferCompleted { handoff_id })
        .unwrap()
        .pop()
        .unwrap();
    source
        .send(BootstrapMessage::Action(activate))
        .await
        .unwrap();
    let envelope = command_rx.recv().await.unwrap();
    assert!(matches!(
        envelope.command,
        SchedulerCommand::ActivateDestination {
            handoff_id: observed,
        } if observed == handoff_id
    ));
    envelope
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::Applied,
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();
    assert!(matches!(
        source.recv().await.unwrap(),
        Some(BootstrapMessage::ActionAck {
            action_id,
            outcome: HandoffActionOutcome::Applied,
        }) if action_id == activate.id
    ));
    let release = coordinator
        .on_action_outcome(activate.id, HandoffActionOutcome::Applied)
        .unwrap()
        .pop()
        .unwrap();
    let complete = coordinator
        .on_action_outcome(release.id, HandoffActionOutcome::Applied)
        .unwrap()
        .pop()
        .unwrap();
    assert!(matches!(complete.action, HandoffAction::Complete { .. }));
    assert!(coordinator.is_complete());
    assert_eq!(coordinator.completion(), Some(HandoffCompletion::Success));

    source.send(BootstrapMessage::Complete).await.unwrap();
    session.await.unwrap().unwrap();
    assert!(registry.register(handoff_id).is_ok());
    shutdown.cancel();
    server.wait_closed().await;
}

#[tokio::test]
async fn premature_complete_keeps_destination_cleanup_active() {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let request_id = Uuid::from_u128(70_100);
    let handoff_id = HandoffId::from(Uuid::from_u128(70_101));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 19,
        request_id,
    };
    let destination = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::DestinationFirst,
            engine_type: EngineType::Sglang,
        },
    )
    .await
    .unwrap();
    let mut source = incoming.recv().await.unwrap().connection;
    let registry = HandoffEventRegistry::default();
    let (command_tx, mut command_rx) = mpsc::channel(1);
    let session = tokio::spawn(run_destination_session(
        destination,
        request(request_id, 2),
        command_tx,
        registry.register(handoff_id).unwrap(),
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
    let reserve_command = command_rx.recv().await.unwrap();
    reserve_command
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::DestinationAccepted { request_id },
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();
    registry
        .deliver_and_wait(SchedulerLifecycleEvent::DestinationReserved {
            handoff_id,
            request_id,
            transferable_prompt_tokens: 1,
        })
        .await;
    assert!(matches!(
        source.recv().await.unwrap(),
        Some(BootstrapMessage::ActionAck {
            action_id,
            outcome: HandoffActionOutcome::Accepted,
        }) if action_id == reserve.id
    ));
    assert!(matches!(
        source.recv().await.unwrap(),
        Some(BootstrapMessage::Fact(
            HandoffFact::DestinationReserved { .. }
        ))
    ));

    coordinator
        .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
        .unwrap();
    let submit = coordinator
        .on_fact(HandoffFact::DestinationReserved {
            handoff_id,
            transferable_prompt_tokens: 1,
        })
        .unwrap()
        .pop()
        .unwrap();
    coordinator
        .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
        .unwrap();
    let transfer = coordinator
        .on_fact(HandoffFact::SourceHeld {
            handoff_id,
            transfer_timing: transfer_timing(None),
        })
        .unwrap()
        .pop()
        .unwrap();
    coordinator
        .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
        .unwrap();
    let activate = coordinator
        .on_fact(HandoffFact::TransferCompleted { handoff_id })
        .unwrap()
        .pop()
        .unwrap();
    source
        .send(BootstrapMessage::Action(activate))
        .await
        .unwrap();
    let pending_activation = command_rx.recv().await.unwrap();
    assert!(matches!(
        pending_activation.command,
        SchedulerCommand::ActivateDestination { handoff_id: observed }
            if observed == handoff_id
    ));

    source.send(BootstrapMessage::Complete).await.unwrap();
    let cleanup = tokio::time::timeout(Duration::from_secs(1), command_rx.recv())
        .await
        .expect("premature completion must trigger destination cleanup")
        .unwrap();
    assert!(matches!(
        cleanup.command,
        SchedulerCommand::CancelDestination { handoff_id: observed }
            if observed == handoff_id
    ));
    cleanup
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::Applied,
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();
    drop(pending_activation);

    let error = session.await.unwrap().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("source completed before destination activation finished")
    );
    shutdown.cancel();
    server.wait_closed().await;
}

#[tokio::test]
async fn source_held_waits_for_submit_outcome_before_progressing() {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let request_id = Uuid::from_u128(71_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(71_001));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 18,
        request_id,
    };
    let mut destination = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity.clone(),
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    let source_connection = incoming.recv().await.unwrap().connection;
    let registry = HandoffEventRegistry::default();
    let route = registry.register(handoff_id).unwrap();
    let (command_tx, mut command_rx) = mpsc::channel(2);
    let (completion_tx, completion_rx) = oneshot::channel();
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let source = SourceRegistration {
        identity,
        order: HandoffOrder::SourceFirst,
        engine_type: EngineType::Vllm,
        request: request(request_id, 1),
        command_tx,
        lifecycle: route,
        completion_tx,
        cancel: CancellationToken::new(),
        observer: None,
        _permit: permits.clone().try_acquire_owned().unwrap(),
    };
    let session = tokio::spawn(run_source_session(
        source,
        source_connection,
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Registered)
    ));
    let submit = command_rx.recv().await.unwrap();
    assert!(matches!(
        submit.command,
        SchedulerCommand::SubmitHandoffPrefill {
            handoff_id: observed,
            ..
        } if observed == handoff_id
    ));

    registry
        .deliver_and_wait(SchedulerLifecycleEvent::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing: transfer_timing(Some(0.0)),
        })
        .await;
    submit
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::Submitted(request_id),
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();

    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Fact(HandoffFact::SourceHeld {
            handoff_id: observed,
            transfer_timing: observed_timing,
        })) if observed == handoff_id
            && observed_timing == transfer_timing(Some(0.0))
    ));
    let reserve = match destination.recv().await.unwrap().unwrap() {
        BootstrapMessage::Action(
            action @ IssuedHandoffAction {
                action: HandoffAction::ReserveDestination { .. },
                ..
            },
        ) => action,
        other => panic!("expected destination reservation, got {other:?}"),
    };
    destination
        .send(BootstrapMessage::ActionAck {
            action_id: reserve.id,
            outcome: HandoffActionOutcome::Accepted,
        })
        .await
        .unwrap();
    destination
        .send(BootstrapMessage::Fact(HandoffFact::DestinationReserved {
            handoff_id,
            transferable_prompt_tokens: 1,
        }))
        .await
        .unwrap();

    let activate = match destination.recv().await.unwrap().unwrap() {
        BootstrapMessage::Action(
            action @ IssuedHandoffAction {
                action: HandoffAction::ActivateDestination { .. },
                ..
            },
        ) => action,
        other => panic!("expected destination activation, got {other:?}"),
    };
    destination
        .send(BootstrapMessage::ActionAck {
            action_id: activate.id,
            outcome: HandoffActionOutcome::Applied,
        })
        .await
        .unwrap();

    let release = command_rx.recv().await.unwrap();
    assert!(matches!(
        release.command,
        SchedulerCommand::ReleaseSource {
            handoff_id: observed,
        } if observed == handoff_id
    ));
    release
        .reply
        .send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::Applied,
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }))
        .unwrap();
    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Complete)
    ));

    session.await.unwrap().unwrap();
    completion_rx.await.unwrap().unwrap();
    assert_eq!(permits.available_permits(), 1);
    drop(registry.register(handoff_id).unwrap());
    shutdown.cancel();
    server.wait_closed().await;
}

enum SourceTerminalTrigger {
    Cancel,
    CloseLifecycle,
}

async fn run_source_terminal_trigger(trigger: SourceTerminalTrigger, id: u128) -> String {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let request_id = Uuid::from_u128(id);
    let handoff_id = HandoffId::from(Uuid::from_u128(id + 1));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 19,
        request_id,
    };
    let mut destination = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity.clone(),
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    let source_connection = incoming.recv().await.unwrap().connection;
    let registry = HandoffEventRegistry::default();
    let route = registry.register(handoff_id).unwrap();
    let (command_tx, mut command_rx) = mpsc::channel::<SchedulerCommandEnvelope>(4);
    let command_task = tokio::spawn(async move {
        let mut canceled = false;
        while let Some(envelope) = command_rx.recv().await {
            let result = match envelope.command {
                SchedulerCommand::SubmitHandoffPrefill { .. } => {
                    SchedulerCommandResult::Submitted(request_id)
                }
                SchedulerCommand::CancelSource {
                    handoff_id: observed,
                } if observed == handoff_id => {
                    canceled = true;
                    SchedulerCommandResult::Applied
                }
                _ => panic!("unexpected source command during cleanup"),
            };
            let _ = envelope.reply.send(Ok(SchedulerCommandEffects {
                result,
                lifecycle_events: Vec::new(),
                kv_events: Vec::new(),
            }));
            if canceled {
                break;
            }
        }
        canceled
    });
    let cancel = CancellationToken::new();
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let (completion_tx, completion_rx) = oneshot::channel();
    let source = SourceRegistration {
        identity,
        order: HandoffOrder::SourceFirst,
        engine_type: EngineType::Vllm,
        request: request(request_id, 1),
        command_tx,
        lifecycle: route,
        completion_tx,
        cancel: cancel.clone(),
        observer: None,
        _permit: permits.clone().try_acquire_owned().unwrap(),
    };
    let session = tokio::spawn(run_source_session(
        source,
        source_connection,
        Duration::from_secs(2),
        shutdown.clone(),
    ));
    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::Registered)
    ));

    match trigger {
        SourceTerminalTrigger::Cancel => cancel.cancel(),
        SourceTerminalTrigger::CloseLifecycle => {
            registry.routes.remove(&handoff_id);
        }
    }

    let error = tokio::time::timeout(Duration::from_secs(1), session)
        .await
        .expect("source terminal watcher did not wake")
        .unwrap()
        .unwrap_err()
        .to_string();
    assert!(completion_rx.await.unwrap().is_err());
    assert!(
        command_task.await.unwrap(),
        "source cleanup command was not sent"
    );
    assert_eq!(permits.available_permits(), 1);
    shutdown.cancel();
    server.wait_closed().await;
    error
}

#[tokio::test]
async fn active_source_request_cancellation_runs_exact_cleanup() {
    let error = run_source_terminal_trigger(SourceTerminalTrigger::Cancel, 72_000).await;
    assert!(error.contains("source request was canceled"));
}

#[tokio::test]
async fn source_lifecycle_channel_closure_runs_exact_cleanup() {
    let error = run_source_terminal_trigger(SourceTerminalTrigger::CloseLifecycle, 73_000).await;
    assert!(error.contains("source lifecycle channel closed"));
}

#[derive(Default)]
struct DestinationKvObservation {
    before_activation: usize,
    activation_hashes: Vec<dynamo_kv_router::protocols::ExternalSequenceBlockHash>,
}

fn destination_command_proxy(
    real_command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    mut kv_events: mpsc::UnboundedReceiver<KvCacheEvent>,
) -> (
    mpsc::Sender<SchedulerCommandEnvelope>,
    tokio::task::JoinHandle<(
        DestinationKvObservation,
        mpsc::UnboundedReceiver<KvCacheEvent>,
    )>,
) {
    let (proxy_tx, mut proxy_rx) = mpsc::channel::<SchedulerCommandEnvelope>(8);
    let task = tokio::spawn(async move {
        let mut observation = DestinationKvObservation::default();
        while let Some(SchedulerCommandEnvelope { command, reply }) = proxy_rx.recv().await {
            enum Phase {
                Reserve,
                Activate,
                Other,
            }
            let phase = match &command {
                SchedulerCommand::ReserveDestination { .. } => Phase::Reserve,
                SchedulerCommand::ActivateDestination { .. } => Phase::Activate,
                _ => Phase::Other,
            };
            let (inner_reply, inner_result) = oneshot::channel();
            let result = if real_command_tx
                .send(SchedulerCommandEnvelope {
                    command,
                    reply: inner_reply,
                })
                .await
                .is_err()
            {
                Err(anyhow::anyhow!(
                    "destination scheduler command channel closed"
                ))
            } else {
                inner_result.await.unwrap_or_else(|_| {
                    Err(anyhow::anyhow!(
                        "destination scheduler command reply was dropped"
                    ))
                })
            };
            match phase {
                Phase::Reserve => {
                    observation.before_activation += drain_stored_hashes(&mut kv_events).len();
                }
                Phase::Activate => {
                    observation
                        .activation_hashes
                        .extend(drain_stored_hashes(&mut kv_events));
                }
                Phase::Other => {}
            }
            let _ = reply.send(result);
        }
        (observation, kv_events)
    });
    (proxy_tx, task)
}

async fn run_live_handoff_conformance(
    engine_type: EngineType,
    transfer_timing_mode: KvTransferTimingMode,
    source_arrives_first: bool,
    case: usize,
) -> NormalizedHandoffConformance {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 2,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let incoming = server.take_incoming_receiver().unwrap();
    let manager =
        SourceHandoffManager::start(incoming, 2, Duration::from_secs(2), shutdown.clone());

    let (mut source_scheduler, mut source_output) = start_scheduler_with_mode(
        engine_type,
        WorkerType::Prefill,
        transfer_timing_mode,
        shutdown.clone(),
    );
    let (mut destination_scheduler, mut destination_output, destination_kv_events) =
        start_scheduler_with_kv_events_and_mode(
            engine_type,
            WorkerType::Decode,
            transfer_timing_mode,
            shutdown.clone(),
        );
    let (destination_command_tx, destination_kv_observer) = destination_command_proxy(
        destination_scheduler.command_sender(),
        destination_kv_events,
    );
    let source_metrics = source_scheduler.metrics_receiver();
    let destination_metrics = destination_scheduler.metrics_receiver();
    let source_registry = HandoffEventRegistry::default();
    let destination_registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        source_scheduler.take_lifecycle_receiver().unwrap(),
        source_registry.clone(),
    );
    spawn_lifecycle_dispatcher(
        destination_scheduler.take_lifecycle_receiver().unwrap(),
        destination_registry.clone(),
    );

    let request_id = Uuid::from_u128(60_000 + case as u128);
    let handoff_id = HandoffId::from(Uuid::from_u128(61_000 + case as u128));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 7,
        request_id,
    };
    let order = order_for_engine(engine_type).unwrap();
    let source_route = source_registry.register(handoff_id).unwrap();
    let destination_route = destination_registry.register(handoff_id).unwrap();
    let (completion_tx, mut completion_rx) = oneshot::channel();
    let (observer_tx, mut observer_rx) = mpsc::unbounded_channel();
    let observer = tokio::spawn(async move {
        let mut lifecycle = Vec::new();
        loop {
            let event = observer_rx.recv().await.unwrap();
            lifecycle.push(event);
            if event == NormalizedHandoffEvent::Completed {
                return lifecycle;
            }
        }
    });
    let source_permits = Arc::new(tokio::sync::Semaphore::new(1));
    let mut source_registration = Some(SourceRegistration {
        identity: identity.clone(),
        order,
        engine_type,
        request: request(request_id, 1),
        command_tx: source_scheduler.command_sender(),
        lifecycle: source_route,
        completion_tx,
        cancel: CancellationToken::new(),
        observer: Some(observer_tx),
        _permit: source_permits.clone().try_acquire_owned().unwrap(),
    });
    let destination_registration = ParticipantRegistration {
        role: BootstrapParticipantRole::Destination,
        dp_rank: 0,
        order,
        engine_type,
    };

    if source_arrives_first {
        manager
            .try_register(source_registration.take().unwrap())
            .unwrap();
    }
    let connection = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        destination_registration,
    )
    .await
    .unwrap();
    if !source_arrives_first {
        manager.wait_for_pending_destination(handoff_id).await;
        manager
            .try_register(source_registration.take().unwrap())
            .unwrap();
    }
    let destination_session = tokio::spawn(run_destination_session(
        connection,
        request(request_id, 2),
        destination_command_tx,
        destination_route,
        CancellationToken::new(),
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    let mut source_outputs = observe_through_first_terminal(&mut source_output, request_id).await;
    assert!(matches!(
        completion_rx.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    tokio::time::timeout(Duration::from_secs(2), completion_rx)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    destination_session.await.unwrap().unwrap();
    let mut destination_outputs =
        observe_through_first_terminal(&mut destination_output, request_id).await;
    let lifecycle = observer.await.unwrap();
    let (destination_kv, mut remaining_kv_events) = destination_kv_observer.await.unwrap();
    let activation_hashes = destination_kv.activation_hashes;
    let activation_set = activation_hashes.iter().copied().collect::<HashSet<_>>();
    let repeated_activation_hashes_after_activation = drain_stored_hashes(&mut remaining_kv_events)
        .into_iter()
        .filter(|hash| activation_set.contains(hash))
        .count();

    wait_for_scheduler_idle(source_metrics).await;
    wait_for_scheduler_idle(destination_metrics).await;
    drain_output_observation(&mut source_output, request_id, &mut source_outputs);
    drain_output_observation(
        &mut destination_output,
        request_id,
        &mut destination_outputs,
    );
    let source_drained = send_command(
        &source_scheduler.command_sender(),
        SchedulerCommand::CancelSource { handoff_id },
    )
    .await
    .unwrap()
    .result
        == SchedulerCommandResult::Noop;
    let destination_drained = send_command(
        &destination_scheduler.command_sender(),
        SchedulerCommand::CancelDestination { handoff_id },
    )
    .await
    .unwrap()
    .result
        == SchedulerCommandResult::Noop;
    shutdown.cancel();
    manager.wait_closed().await;
    server.wait_closed().await;
    let driver_drained = source_permits.available_permits() == 1
        && source_registry.register(handoff_id).is_ok()
        && destination_registry.register(handoff_id).is_ok();

    let report = NormalizedHandoffConformance {
        engine_type,
        order,
        lifecycle,
        source_output_tokens: source_outputs.output_tokens,
        destination_output_tokens: destination_outputs.output_tokens,
        completed_requests: destination_outputs.completed_requests,
        destination_stored: NormalizedStoredTiming {
            before_activation: destination_kv.before_activation,
            on_activation: activation_hashes.len(),
            repeated_activation_hashes_after_activation,
        },
        source_drained,
        destination_drained,
        driver_drained,
    };
    assert_eq!(source_outputs.completed_requests, 1);
    report.validate().unwrap();
    report
}

#[tokio::test]
async fn live_and_offline_handoff_surfaces_share_one_conformance_report() {
    let mut case = 0;
    for transfer_timing_mode in [
        KvTransferTimingMode::FullPrompt,
        KvTransferTimingMode::DestinationMissing,
    ] {
        for (engine_type, source_arrives_first) in [
            (EngineType::Vllm, true),
            (EngineType::Vllm, false),
            (EngineType::Sglang, true),
            (EngineType::Sglang, false),
        ] {
            let live = run_live_handoff_conformance(
                engine_type,
                transfer_timing_mode,
                source_arrives_first,
                case,
            )
            .await;
            let offline = dynamo_mocker::replay::run_offline_handoff_conformance(
                engine_type,
                transfer_timing_mode,
            )
            .unwrap();
            assert_eq!(live, offline);
            case += 1;
        }
    }
}

#[tokio::test]
async fn bootstrap_rendezvous_rejects_backend_mismatch_before_scheduler_ownership() {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(0, shutdown.clone(), BootstrapServerConfig::default())
        .await
        .unwrap();
    let incoming = server.take_incoming_receiver().unwrap();
    let manager =
        SourceHandoffManager::start(incoming, 1, Duration::from_secs(2), shutdown.clone());
    let handoff_id = HandoffId::from(Uuid::from_u128(62_000));
    let request_id = Uuid::from_u128(62_001);
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 8,
        request_id,
    };
    let registry = HandoffEventRegistry::default();
    let permit_pool = Arc::new(tokio::sync::Semaphore::new(1));
    let (command_tx, mut command_rx) = mpsc::channel(1);
    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(SourceRegistration {
            identity: identity.clone(),
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            request: request(request_id, 1),
            command_tx,
            lifecycle: registry.register(handoff_id).unwrap(),
            completion_tx,
            cancel: CancellationToken::new(),
            observer: None,
            _permit: permit_pool.clone().try_acquire_owned().unwrap(),
        })
        .unwrap();

    let mut destination = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Sglang,
        },
    )
    .await
    .unwrap();

    let source_error = completion_rx.await.unwrap().unwrap_err();
    assert!(source_error.contains("identity or backend mismatch"));
    assert!(matches!(
        destination.recv().await.unwrap(),
        Some(BootstrapMessage::ProtocolError { message })
            if message.contains("identity or backend mismatch")
    ));
    assert!(command_rx.try_recv().is_err());
    assert_eq!(permit_pool.available_permits(), 1);

    shutdown.cancel();
    manager.wait_closed().await;
    server.wait_closed().await;
}

async fn assert_destination_disconnect_cleans_ownership(
    engine_type: EngineType,
    activate_before_disconnect: bool,
) {
    let shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 1,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let mut incoming_rx = server.take_incoming_receiver().unwrap();
    let (mut destination_scheduler, _output) =
        start_scheduler(engine_type, WorkerType::Decode, shutdown.clone());
    let registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        destination_scheduler.take_lifecycle_receiver().unwrap(),
        registry.clone(),
    );
    let request_id = Uuid::from_u128(70_000 + u128::from(activate_before_disconnect));
    let handoff_id = HandoffId::from(Uuid::from_u128(
        70_010 + u128::from(activate_before_disconnect),
    ));
    let order = order_for_engine(engine_type).unwrap();
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 9,
        request_id,
    };
    let client = connect_to_prefill(
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
    let mut incoming = incoming_rx.recv().await.unwrap();
    let command_tx = destination_scheduler.command_sender();
    let route = registry.register(handoff_id).unwrap();
    let destination = tokio::spawn(run_destination_session(
        client,
        request(request_id, 16),
        command_tx.clone(),
        route,
        CancellationToken::new(),
        Duration::from_secs(2),
        shutdown.clone(),
    ));

    incoming
        .connection
        .send(BootstrapMessage::Registered)
        .await
        .unwrap();
    let mut coordinator = HandoffCoordinatorCore::new(handoff_id, order);
    let reserve = match order {
        HandoffOrder::DestinationFirst => coordinator.start().unwrap().pop().unwrap(),
        HandoffOrder::SourceFirst => {
            let submit = coordinator.start().unwrap().pop().unwrap();
            coordinator
                .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
                .unwrap();
            coordinator
                .on_fact(HandoffFact::SourceHeld {
                    handoff_id,
                    transfer_timing: transfer_timing(None),
                })
                .unwrap()
                .pop()
                .unwrap()
        }
    };
    incoming
        .connection
        .send(BootstrapMessage::Action(reserve))
        .await
        .unwrap();
    let mut accepted = false;
    let mut reserved = false;
    while !accepted || !reserved {
        match incoming.connection.recv().await.unwrap().unwrap() {
            BootstrapMessage::ActionAck { action_id, outcome }
                if action_id == reserve.id && outcome == HandoffActionOutcome::Accepted =>
            {
                accepted = true;
            }
            BootstrapMessage::Fact(HandoffFact::DestinationReserved {
                handoff_id: observed,
                ..
            }) if observed == handoff_id => reserved = true,
            other => panic!("unexpected destination message: {other:?}"),
        }
    }
    if activate_before_disconnect {
        coordinator
            .on_action_outcome(reserve.id, HandoffActionOutcome::Accepted)
            .unwrap();
        let after_reserved = coordinator
            .on_fact(HandoffFact::DestinationReserved {
                handoff_id,
                transferable_prompt_tokens: 1,
            })
            .unwrap();
        let transfer = match order {
            HandoffOrder::SourceFirst => after_reserved.into_iter().next().unwrap(),
            HandoffOrder::DestinationFirst => {
                let submit = after_reserved.into_iter().next().unwrap();
                coordinator
                    .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
                    .unwrap();
                coordinator
                    .on_fact(HandoffFact::SourceHeld {
                        handoff_id,
                        transfer_timing: transfer_timing(None),
                    })
                    .unwrap()
                    .pop()
                    .unwrap()
            }
        };
        coordinator
            .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
            .unwrap();
        let activation = coordinator
            .on_fact(HandoffFact::TransferCompleted { handoff_id })
            .unwrap()
            .pop()
            .unwrap();
        incoming
            .connection
            .send(BootstrapMessage::Action(activation))
            .await
            .unwrap();
        assert!(matches!(
            incoming.connection.recv().await.unwrap(),
            Some(BootstrapMessage::ActionAck {
                action_id,
                outcome: HandoffActionOutcome::Applied,
            }) if action_id == activation.id
        ));
    }
    drop(incoming);

    assert!(destination.await.unwrap().is_err());
    assert_eq!(
        send_command(
            &command_tx,
            SchedulerCommand::CancelDestination { handoff_id },
        )
        .await
        .unwrap()
        .result,
        SchedulerCommandResult::Noop
    );
    shutdown.cancel();
    server.wait_closed().await;
}

#[tokio::test]
async fn destination_disconnect_cleans_reserved_and_activated_ownership() {
    assert_destination_disconnect_cleans_ownership(EngineType::Vllm, false).await;
    assert_destination_disconnect_cleans_ownership(EngineType::Vllm, true).await;
    assert_destination_disconnect_cleans_ownership(EngineType::Sglang, true).await;
}

#[tokio::test]
async fn lost_reserve_or_activation_ack_cleans_ambiguous_destination_ownership() {
    for (case, lose_activation_ack) in [false, true].into_iter().enumerate() {
        let shutdown = CancellationToken::new();
        let server = BootstrapServer::start(
            0,
            shutdown.clone(),
            BootstrapServerConfig {
                max_pending_connections: 1,
                ..BootstrapServerConfig::default()
            },
        )
        .await
        .unwrap();
        let mut incoming_rx = server.take_incoming_receiver().unwrap();
        let (mut destination_scheduler, _output) =
            start_scheduler(EngineType::Vllm, WorkerType::Decode, shutdown.clone());
        let registry = HandoffEventRegistry::default();
        spawn_lifecycle_dispatcher(
            destination_scheduler.take_lifecycle_receiver().unwrap(),
            registry.clone(),
        );
        let request_id = Uuid::from_u128(72_000 + case as u128);
        let handoff_id = HandoffId::from(Uuid::from_u128(72_100 + case as u128));
        let identity = BootstrapIdentity {
            handoff_id,
            bootstrap_room: 14 + case as u64,
            request_id,
        };
        let client = connect_to_prefill(
            "127.0.0.1",
            server.port(),
            identity,
            ParticipantRegistration {
                role: BootstrapParticipantRole::Destination,
                dp_rank: 0,
                order: HandoffOrder::SourceFirst,
                engine_type: EngineType::Vllm,
            },
        )
        .await
        .unwrap();
        let mut source = incoming_rx.recv().await.unwrap();
        let command_tx = destination_scheduler.command_sender();
        let destination = tokio::spawn(run_destination_session(
            client,
            request(request_id, 2),
            command_tx.clone(),
            registry.register(handoff_id).unwrap(),
            CancellationToken::new(),
            Duration::from_secs(2),
            shutdown.clone(),
        ));
        source
            .connection
            .send(BootstrapMessage::Registered)
            .await
            .unwrap();

        let mut coordinator = HandoffCoordinatorCore::new(handoff_id, HandoffOrder::SourceFirst);
        let submit = coordinator.start().unwrap().pop().unwrap();
        coordinator
            .on_action_outcome(submit.id, HandoffActionOutcome::Submitted)
            .unwrap();
        let reserve = coordinator
            .on_fact(HandoffFact::SourceHeld {
                handoff_id,
                transfer_timing: transfer_timing(None),
            })
            .unwrap()
            .pop()
            .unwrap();
        source
            .connection
            .send(BootstrapMessage::Action(reserve))
            .await
            .unwrap();

        if lose_activation_ack {
            let reserve_ack = source.connection.recv().await.unwrap().unwrap();
            let BootstrapMessage::ActionAck { action_id, outcome } = reserve_ack else {
                panic!("expected reserve acknowledgement");
            };
            coordinator.on_action_outcome(action_id, outcome).unwrap();
            assert!(matches!(
                source.connection.recv().await.unwrap(),
                Some(BootstrapMessage::Fact(
                    HandoffFact::DestinationReserved { .. }
                ))
            ));
            let transfer = coordinator
                .on_fact(HandoffFact::DestinationReserved {
                    handoff_id,
                    transferable_prompt_tokens: 1,
                })
                .unwrap()
                .pop()
                .unwrap();
            coordinator
                .on_action_outcome(transfer.id, HandoffActionOutcome::Scheduled)
                .unwrap();
            let activation = coordinator
                .on_fact(HandoffFact::TransferCompleted { handoff_id })
                .unwrap()
                .pop()
                .unwrap();
            source
                .connection
                .send(BootstrapMessage::Action(activation))
                .await
                .unwrap();
        }
        drop(source);

        assert!(destination.await.unwrap().is_err());
        assert_eq!(
            send_command(
                &command_tx,
                SchedulerCommand::CancelDestination { handoff_id },
            )
            .await
            .unwrap()
            .result,
            SchedulerCommandResult::Noop
        );
        shutdown.cancel();
        server.wait_closed().await;
    }
}

#[tokio::test(start_paused = true)]
async fn cleanup_ack_can_wait_beyond_the_rendezvous_timeout() {
    let (command_tx, mut command_rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let envelope: SchedulerCommandEnvelope = command_rx.recv().await.unwrap();
        tokio::time::sleep(Duration::from_secs(31)).await;
        let _ = envelope.reply.send(Ok(SchedulerCommandEffects {
            result: SchedulerCommandResult::Noop,
            lifecycle_events: Vec::new(),
            kv_events: Vec::new(),
        }));
    });

    let effects = send_cleanup_command(
        &command_tx,
        SchedulerCommand::CancelSource {
            handoff_id: HandoffId::new(),
        },
        tokio::time::Instant::now() + Duration::from_secs(40),
    )
    .await
    .unwrap();
    assert_eq!(effects.result, SchedulerCommandResult::Noop);
}

#[tokio::test]
async fn duplicate_lifecycle_route_does_not_replace_the_active_route() {
    let registry = HandoffEventRegistry::default();
    let handoff_id = HandoffId::from(Uuid::from_u128(80_000));
    let request_id = Uuid::from_u128(80_001);
    let mut route = registry.register(handoff_id).unwrap();

    assert!(registry.register(handoff_id).is_err());
    registry
        .deliver(SchedulerLifecycleEvent::SourceHeld {
            handoff_id,
            request_id,
            transfer_timing: transfer_timing(None),
        })
        .await;

    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), route.recv())
            .await
            .unwrap(),
        Some(SchedulerLifecycleEvent::SourceHeld {
            handoff_id: observed,
            request_id: observed_request,
            ..
        }) if observed == handoff_id && observed_request == request_id
    ));
}

#[tokio::test]
async fn pending_source_cancellation_releases_session_capacity_before_rendezvous_timeout() {
    let (_incoming_tx, incoming_rx) = mpsc::channel(1);
    let shutdown = CancellationToken::new();
    let manager =
        SourceHandoffManager::start(incoming_rx, 1, Duration::from_secs(1), shutdown.clone());
    let registry = HandoffEventRegistry::default();
    let handoff_id = HandoffId::from(Uuid::from_u128(81_000));
    let cancel = CancellationToken::new();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
    let permit = semaphore.clone().try_acquire_owned().unwrap();
    let (command_tx, _command_rx) = mpsc::channel(1);
    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(SourceRegistration {
            identity: BootstrapIdentity {
                handoff_id,
                bootstrap_room: 11,
                request_id: Uuid::from_u128(81_001),
            },
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            request: request(Uuid::from_u128(81_001), 1),
            command_tx,
            lifecycle: registry.register(handoff_id).unwrap(),
            completion_tx,
            cancel: cancel.clone(),
            observer: None,
            _permit: permit,
        })
        .unwrap();

    manager.wait_for_pending_source(handoff_id).await;
    cancel.cancel();
    let error = completion_rx.await.unwrap().unwrap_err();
    assert!(error.contains("canceled before rendezvous"));
    assert_eq!(semaphore.available_permits(), 1);
    shutdown.cancel();
    manager.wait_closed().await;
}

#[tokio::test(start_paused = true)]
async fn rendezvous_timeout_retires_the_session_id() {
    let (_incoming_tx, incoming_rx) = mpsc::channel(1);
    let shutdown = CancellationToken::new();
    let manager = SourceHandoffManager::start_with_rendezvous_timeout(
        incoming_rx,
        1,
        Duration::from_secs(10),
        Duration::from_secs(1),
        shutdown.clone(),
    );
    let registry = HandoffEventRegistry::default();
    let handoff_id = HandoffId::from(Uuid::from_u128(81_500));
    let permit_pool = Arc::new(tokio::sync::Semaphore::new(1));
    let (command_tx, _command_rx) = mpsc::channel(1);
    let make_registration = |completion_tx| SourceRegistration {
        identity: BootstrapIdentity {
            handoff_id,
            bootstrap_room: 15,
            request_id: Uuid::from_u128(81_501),
        },
        order: HandoffOrder::SourceFirst,
        engine_type: EngineType::Vllm,
        request: request(Uuid::from_u128(81_501), 1),
        command_tx: command_tx.clone(),
        lifecycle: registry.register(handoff_id).unwrap(),
        completion_tx,
        cancel: CancellationToken::new(),
        observer: None,
        _permit: permit_pool.clone().try_acquire_owned().unwrap(),
    };

    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(make_registration(completion_tx))
        .unwrap();
    let error = completion_rx.await.unwrap().unwrap_err();
    assert!(error.contains("rendezvous timed out"));
    tokio::task::yield_now().await;
    assert_eq!(permit_pool.available_permits(), 1);

    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(make_registration(completion_tx))
        .unwrap();
    let error = completion_rx.await.unwrap().unwrap_err();
    assert!(error.contains("retired"));
    tokio::task::yield_now().await;
    assert_eq!(permit_pool.available_permits(), 1);

    shutdown.cancel();
    manager.wait_closed().await;
}

#[tokio::test]
async fn pending_destination_disconnect_is_retired_without_rendezvous_timeout() {
    let manager_shutdown = CancellationToken::new();
    let scheduler_shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        manager_shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 1,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let incoming = server.take_incoming_receiver().unwrap();
    let manager = SourceHandoffManager::start(
        incoming,
        1,
        Duration::from_secs(2),
        manager_shutdown.clone(),
    );
    let (mut source_scheduler, _output) = start_scheduler(
        EngineType::Vllm,
        WorkerType::Prefill,
        scheduler_shutdown.clone(),
    );
    let registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        source_scheduler.take_lifecycle_receiver().unwrap(),
        registry.clone(),
    );
    let request_id = Uuid::from_u128(82_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(82_001));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 12,
        request_id,
    };
    let client = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity.clone(),
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    manager.wait_for_pending_destination(handoff_id).await;
    drop(client);
    manager.wait_for_retired(handoff_id).await;

    let permit_pool = Arc::new(tokio::sync::Semaphore::new(1));
    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(SourceRegistration {
            identity,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            request: request(request_id, 1),
            command_tx: source_scheduler.command_sender(),
            lifecycle: registry.register(handoff_id).unwrap(),
            completion_tx,
            cancel: CancellationToken::new(),
            observer: None,
            _permit: permit_pool.clone().try_acquire_owned().unwrap(),
        })
        .unwrap();
    let error = completion_rx.await.unwrap().unwrap_err();
    assert!(error.contains("retired"));
    assert_eq!(permit_pool.available_permits(), 1);

    manager_shutdown.cancel();
    manager.wait_closed().await;
    server.wait_closed().await;
    scheduler_shutdown.cancel();
}

#[tokio::test]
async fn active_shutdown_joins_sessions_after_exact_cleanup() {
    let manager_shutdown = CancellationToken::new();
    let scheduler_shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        manager_shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 1,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let incoming = server.take_incoming_receiver().unwrap();
    let manager = SourceHandoffManager::start(
        incoming,
        1,
        Duration::from_secs(2),
        manager_shutdown.clone(),
    );
    let (mut source_scheduler, _source_output) = start_scheduler(
        EngineType::Vllm,
        WorkerType::Prefill,
        scheduler_shutdown.clone(),
    );
    let (mut destination_scheduler, _destination_output) = start_scheduler(
        EngineType::Vllm,
        WorkerType::Decode,
        scheduler_shutdown.clone(),
    );
    let source_registry = HandoffEventRegistry::default();
    let destination_registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        source_scheduler.take_lifecycle_receiver().unwrap(),
        source_registry.clone(),
    );
    spawn_lifecycle_dispatcher(
        destination_scheduler.take_lifecycle_receiver().unwrap(),
        destination_registry.clone(),
    );
    let request_id = Uuid::from_u128(83_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(83_001));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 13,
        request_id,
    };
    let permit_pool = Arc::new(tokio::sync::Semaphore::new(1));
    let (completion_tx, completion_rx) = oneshot::channel();
    manager
        .try_register(SourceRegistration {
            identity: identity.clone(),
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
            request: request(request_id, 1),
            command_tx: source_scheduler.command_sender(),
            lifecycle: source_registry.register(handoff_id).unwrap(),
            completion_tx,
            cancel: CancellationToken::new(),
            observer: None,
            _permit: permit_pool.clone().try_acquire_owned().unwrap(),
        })
        .unwrap();
    let connection = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    let source_command_tx = source_scheduler.command_sender();
    let destination_command_tx = destination_scheduler.command_sender();
    let destination = tokio::spawn(run_destination_session(
        connection,
        request(request_id, 2),
        destination_command_tx.clone(),
        destination_registry.register(handoff_id).unwrap(),
        CancellationToken::new(),
        Duration::from_secs(2),
        manager_shutdown.clone(),
    ));
    manager.wait_for_active(handoff_id).await;
    manager_shutdown.cancel();

    manager.wait_closed().await;
    server.wait_closed().await;
    assert!(completion_rx.await.unwrap().is_err());
    assert!(destination.await.unwrap().is_err());
    assert_eq!(permit_pool.available_permits(), 1);
    assert_eq!(
        send_command(
            &source_command_tx,
            SchedulerCommand::CancelSource { handoff_id },
        )
        .await
        .unwrap()
        .result,
        SchedulerCommandResult::Noop
    );
    assert_eq!(
        send_command(
            &destination_command_tx,
            SchedulerCommand::CancelDestination { handoff_id },
        )
        .await
        .unwrap()
        .result,
        SchedulerCommandResult::Noop
    );
    assert!(source_registry.register(handoff_id).is_ok());
    assert!(destination_registry.register(handoff_id).is_ok());
    scheduler_shutdown.cancel();
}

#[tokio::test]
async fn expired_source_session_deadline_still_cleans_held_ownership() {
    let transport_shutdown = CancellationToken::new();
    let scheduler_shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        transport_shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 1,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let (mut source_scheduler, _output) = start_scheduler(
        EngineType::Vllm,
        WorkerType::Prefill,
        scheduler_shutdown.clone(),
    );
    let registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        source_scheduler.take_lifecycle_receiver().unwrap(),
        registry.clone(),
    );
    let request_id = Uuid::from_u128(84_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(84_001));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 14,
        request_id,
    };
    let mut destination_connection = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity.clone(),
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::SourceFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    let source_connection = incoming.recv().await.unwrap().connection;
    let source_command_tx = source_scheduler.command_sender();
    let (completion_tx, completion_rx) = oneshot::channel();
    let source = SourceRegistration {
        identity,
        order: HandoffOrder::SourceFirst,
        engine_type: EngineType::Vllm,
        request: request(request_id, 1),
        command_tx: source_command_tx.clone(),
        lifecycle: registry.register(handoff_id).unwrap(),
        completion_tx,
        cancel: CancellationToken::new(),
        observer: None,
        _permit: Arc::new(tokio::sync::Semaphore::new(1))
            .try_acquire_owned()
            .unwrap(),
    };
    let source_task = tokio::spawn(run_source_session(
        source,
        source_connection,
        Duration::from_millis(100),
        transport_shutdown.clone(),
    ));

    assert!(matches!(
        destination_connection.recv().await.unwrap(),
        Some(BootstrapMessage::Registered)
    ));
    loop {
        let message = destination_connection.recv().await.unwrap().unwrap();
        if matches!(
            message,
            BootstrapMessage::Action(IssuedHandoffAction {
                action: HandoffAction::ReserveDestination { .. },
                ..
            })
        ) {
            break;
        }
    }

    assert!(
        tokio::time::timeout(Duration::from_secs(2), source_task)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    assert!(completion_rx.await.unwrap().is_err());
    assert_eq!(
        send_command(
            &source_command_tx,
            SchedulerCommand::CancelSource { handoff_id },
        )
        .await
        .unwrap()
        .result,
        SchedulerCommandResult::Noop
    );

    transport_shutdown.cancel();
    server.wait_closed().await;
    scheduler_shutdown.cancel();
}

#[tokio::test]
async fn expired_destination_session_deadline_still_cleans_reserved_ownership() {
    let transport_shutdown = CancellationToken::new();
    let scheduler_shutdown = CancellationToken::new();
    let server = BootstrapServer::start(
        0,
        transport_shutdown.clone(),
        BootstrapServerConfig {
            max_pending_connections: 1,
            ..BootstrapServerConfig::default()
        },
    )
    .await
    .unwrap();
    let mut incoming = server.take_incoming_receiver().unwrap();
    let (mut destination_scheduler, _output) = start_scheduler(
        EngineType::Vllm,
        WorkerType::Decode,
        scheduler_shutdown.clone(),
    );
    let registry = HandoffEventRegistry::default();
    spawn_lifecycle_dispatcher(
        destination_scheduler.take_lifecycle_receiver().unwrap(),
        registry.clone(),
    );
    let request_id = Uuid::from_u128(85_000);
    let handoff_id = HandoffId::from(Uuid::from_u128(85_001));
    let identity = BootstrapIdentity {
        handoff_id,
        bootstrap_room: 15,
        request_id,
    };
    let destination_connection = connect_to_prefill(
        "127.0.0.1",
        server.port(),
        identity,
        ParticipantRegistration {
            role: BootstrapParticipantRole::Destination,
            dp_rank: 0,
            order: HandoffOrder::DestinationFirst,
            engine_type: EngineType::Vllm,
        },
    )
    .await
    .unwrap();
    let mut source_connection = incoming.recv().await.unwrap().connection;
    let destination_command_tx = destination_scheduler.command_sender();
    let destination_task = tokio::spawn(run_destination_session(
        destination_connection,
        request(request_id, 2),
        destination_command_tx.clone(),
        registry.register(handoff_id).unwrap(),
        CancellationToken::new(),
        Duration::from_millis(100),
        transport_shutdown.clone(),
    ));

    source_connection
        .send(BootstrapMessage::Registered)
        .await
        .unwrap();
    let mut coordinator = HandoffCoordinatorCore::new(handoff_id, HandoffOrder::DestinationFirst);
    let reserve = coordinator.start().unwrap().pop().unwrap();
    source_connection
        .send(BootstrapMessage::Action(reserve))
        .await
        .unwrap();
    let mut accepted = false;
    let mut reserved = false;
    while !accepted || !reserved {
        match source_connection.recv().await.unwrap().unwrap() {
            BootstrapMessage::ActionAck { action_id, outcome }
                if action_id == reserve.id && outcome == HandoffActionOutcome::Accepted =>
            {
                accepted = true;
            }
            BootstrapMessage::Fact(HandoffFact::DestinationReserved {
                handoff_id: observed,
                ..
            }) if observed == handoff_id => reserved = true,
            other => panic!("unexpected destination message: {other:?}"),
        }
    }

    assert!(
        tokio::time::timeout(Duration::from_secs(2), destination_task)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    assert_eq!(
        send_command(
            &destination_command_tx,
            SchedulerCommand::CancelDestination { handoff_id },
        )
        .await
        .unwrap()
        .result,
        SchedulerCommandResult::Noop
    );

    transport_shutdown.cancel();
    server.wait_closed().await;
    scheduler_shutdown.cancel();
}
