// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use dynamo_mocker::common::handoff::{
    HandoffAction, HandoffActionId, HandoffActionOutcome, HandoffCoordinatorCore, HandoffFact,
    HandoffId, HandoffOrder, HandoffTransferTiming, IssuedHandoffAction, NormalizedHandoffEvent,
    validate_transfer_delay_ms, validate_transfer_timing,
};
use dynamo_mocker::common::protocols::{DirectRequest, EngineType, KvTransferTimingMode};
use dynamo_mocker::scheduler::{
    SchedulerCommand, SchedulerCommandEffects, SchedulerCommandEnvelope, SchedulerCommandResult,
    SchedulerLifecycleEvent,
};
use dynamo_mocker::services::bootstrap::{
    BootstrapConnection, BootstrapIdentity, BootstrapMessage, IncomingBootstrapConnection,
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

const SESSION_INBOX_CAPACITY: usize = 32;
const PARTICIPANT_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

fn session_deadline_with_transfer(
    session_started: tokio::time::Instant,
    session_timeout: Duration,
    transfer_delay_ms: Option<f64>,
) -> Result<tokio::time::Instant> {
    validate_transfer_delay_ms(transfer_delay_ms)?;
    Ok(session_started
        + session_timeout
        + Duration::from_secs_f64(transfer_delay_ms.unwrap_or_default() / 1000.0))
}

fn transfer_timeout_delay(
    transfer_timing: HandoffTransferTiming,
    destination_missing_tokens: Option<usize>,
) -> Option<Option<f64>> {
    match transfer_timing.mode {
        KvTransferTimingMode::FullPrompt => Some(transfer_timing.full_prompt_delay_ms()),
        KvTransferTimingMode::DestinationMissing => {
            destination_missing_tokens.map(|tokens| transfer_timing.delay_ms(tokens))
        }
    }
}

pub(crate) fn order_for_engine(engine_type: EngineType) -> Result<HandoffOrder> {
    match engine_type {
        EngineType::Vllm => Ok(HandoffOrder::SourceFirst),
        EngineType::Sglang => Ok(HandoffOrder::DestinationFirst),
        EngineType::Trtllm => bail!("mocker handoff does not support TRT-LLM"),
    }
}

#[derive(Clone, Default)]
pub(crate) struct HandoffEventRegistry {
    routes: Arc<DashMap<HandoffId, mpsc::Sender<HandoffEventEnvelope>>>,
}

struct HandoffEventEnvelope {
    event: SchedulerLifecycleEvent,
    #[cfg(test)]
    received: Option<oneshot::Sender<()>>,
}

impl HandoffEventRegistry {
    pub(crate) fn register(&self, handoff_id: HandoffId) -> Result<HandoffEventRoute> {
        let (tx, rx) = mpsc::channel(SESSION_INBOX_CAPACITY);
        match self.routes.entry(handoff_id) {
            Entry::Vacant(entry) => {
                entry.insert(tx);
            }
            Entry::Occupied(_) => {
                bail!("handoff {handoff_id:?} already has a lifecycle route");
            }
        }
        Ok(HandoffEventRoute {
            handoff_id,
            routes: self.routes.clone(),
            rx,
        })
    }

    pub(crate) async fn deliver(&self, event: SchedulerLifecycleEvent) {
        let handoff_id = event.handoff_id();
        let sender = self.routes.get(&handoff_id).map(|entry| entry.clone());
        if let Some(sender) = sender {
            let _ = sender
                .send(HandoffEventEnvelope {
                    event,
                    #[cfg(test)]
                    received: None,
                })
                .await;
        }
    }

    #[cfg(test)]
    async fn deliver_and_wait(&self, event: SchedulerLifecycleEvent) {
        let handoff_id = event.handoff_id();
        let Some(sender) = self.routes.get(&handoff_id).map(|entry| entry.clone()) else {
            return;
        };
        let (received, wait) = oneshot::channel();
        if sender
            .send(HandoffEventEnvelope {
                event,
                received: Some(received),
            })
            .await
            .is_ok()
        {
            let _ = wait.await;
        }
    }
}

pub(crate) struct HandoffEventRoute {
    handoff_id: HandoffId,
    routes: Arc<DashMap<HandoffId, mpsc::Sender<HandoffEventEnvelope>>>,
    rx: mpsc::Receiver<HandoffEventEnvelope>,
}

impl HandoffEventRoute {
    async fn recv(&mut self) -> Option<SchedulerLifecycleEvent> {
        let envelope = self.rx.recv().await?;
        #[cfg(test)]
        if let Some(received) = envelope.received {
            let _ = received.send(());
        }
        Some(envelope.event)
    }
}

impl Drop for HandoffEventRoute {
    fn drop(&mut self) {
        self.routes.remove(&self.handoff_id);
    }
}

pub(crate) struct SourceRegistration {
    pub identity: BootstrapIdentity,
    pub order: HandoffOrder,
    pub engine_type: EngineType,
    pub request: DirectRequest,
    pub command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    pub lifecycle: HandoffEventRoute,
    pub completion_tx: oneshot::Sender<Result<(), String>>,
    pub cancel: CancellationToken,
    pub observer: Option<mpsc::UnboundedSender<NormalizedHandoffEvent>>,
    pub _permit: OwnedSemaphorePermit,
}

#[derive(Clone)]
pub(crate) struct SourceHandoffManager {
    source_tx: mpsc::Sender<SourceRegistration>,
    closed_rx: watch::Receiver<bool>,
    #[cfg(test)]
    state_rx: watch::Receiver<SourceHandoffManagerTestState>,
}

#[cfg(test)]
#[derive(Clone, Default)]
struct SourceHandoffManagerTestState {
    pending_sources: HashSet<HandoffId>,
    pending_destinations: HashSet<HandoffId>,
    active: HashSet<HandoffId>,
    retired: HashSet<HandoffId>,
}

impl SourceHandoffManager {
    pub(crate) fn start(
        incoming_rx: mpsc::Receiver<IncomingBootstrapConnection>,
        max_sessions: usize,
        session_timeout: Duration,
        shutdown: CancellationToken,
    ) -> Self {
        Self::start_with_rendezvous_timeout(
            incoming_rx,
            max_sessions,
            session_timeout,
            PARTICIPANT_RENDEZVOUS_TIMEOUT,
            shutdown,
        )
    }

    fn start_with_rendezvous_timeout(
        incoming_rx: mpsc::Receiver<IncomingBootstrapConnection>,
        max_sessions: usize,
        session_timeout: Duration,
        rendezvous_timeout: Duration,
        shutdown: CancellationToken,
    ) -> Self {
        let (source_tx, source_rx) = mpsc::channel(max_sessions.max(1));
        let (closed_tx, closed_rx) = watch::channel(false);
        #[cfg(test)]
        let (state_tx, state_rx) = watch::channel(SourceHandoffManagerTestState::default());
        tokio::spawn(async move {
            run_manager(
                source_rx,
                incoming_rx,
                max_sessions.max(1),
                session_timeout,
                rendezvous_timeout,
                shutdown,
                #[cfg(test)]
                state_tx,
            )
            .await;
            let _ = closed_tx.send(true);
        });
        Self {
            source_tx,
            closed_rx,
            #[cfg(test)]
            state_rx,
        }
    }

    pub(crate) fn try_register(&self, registration: SourceRegistration) -> Result<()> {
        self.source_tx
            .try_send(registration)
            .map_err(|error| anyhow!("mocker handoff source admission failed: {error}"))
    }

    pub(crate) async fn wait_closed(&self) {
        let mut closed_rx = self.closed_rx.clone();
        if *closed_rx.borrow() {
            return;
        }
        let _ = closed_rx.wait_for(|closed| *closed).await;
    }

    #[cfg(test)]
    async fn wait_for_pending_source(&self, handoff_id: HandoffId) {
        let mut state = self.state_rx.clone();
        let _ = state
            .wait_for(|state| state.pending_sources.contains(&handoff_id))
            .await;
    }

    #[cfg(test)]
    async fn wait_for_pending_destination(&self, handoff_id: HandoffId) {
        let mut state = self.state_rx.clone();
        let _ = state
            .wait_for(|state| state.pending_destinations.contains(&handoff_id))
            .await;
    }

    #[cfg(test)]
    async fn wait_for_active(&self, handoff_id: HandoffId) {
        let mut state = self.state_rx.clone();
        let _ = state
            .wait_for(|state| state.active.contains(&handoff_id))
            .await;
    }

    #[cfg(test)]
    async fn wait_for_retired(&self, handoff_id: HandoffId) {
        let mut state = self.state_rx.clone();
        let _ = state
            .wait_for(|state| state.retired.contains(&handoff_id))
            .await;
    }
}

struct PendingSession {
    created_at: tokio::time::Instant,
    source: Option<SourceRegistration>,
    destination: Option<IncomingBootstrapConnection>,
}

async fn run_manager(
    mut source_rx: mpsc::Receiver<SourceRegistration>,
    mut incoming_rx: mpsc::Receiver<IncomingBootstrapConnection>,
    max_sessions: usize,
    session_timeout: Duration,
    rendezvous_timeout: Duration,
    shutdown: CancellationToken,
    #[cfg(test)] state_tx: watch::Sender<SourceHandoffManagerTestState>,
) {
    let mut pending = HashMap::<HandoffId, PendingSession>::new();
    let mut active = HashSet::<HandoffId>::new();
    let mut retired = HashMap::<HandoffId, Instant>::new();
    let mut retired_order = VecDeque::<HandoffId>::new();
    let (done_tx, mut done_rx) = mpsc::channel::<HandoffId>(max_sessions);
    let sessions = TaskTracker::new();
    let rejection_permits = Arc::new(tokio::sync::Semaphore::new(max_sessions));
    let mut cleanup = tokio::time::interval(Duration::from_millis(100));

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            Some(handoff_id) = done_rx.recv() => {
                active.remove(&handoff_id);
                retire_session(
                    handoff_id,
                    session_timeout,
                    max_sessions,
                    &mut retired,
                    &mut retired_order,
                );
            }
            Some(source) = source_rx.recv() => {
                let handoff_id = source.identity.handoff_id;
                prune_retired(&mut retired, &mut retired_order);
                if retired.contains_key(&handoff_id) || active.contains(&handoff_id) {
                    reject_source(source, "handoff session is already active or retired");
                    continue;
                }
                if !pending.contains_key(&handoff_id) && pending.len() + active.len() >= max_sessions {
                    reject_source(source, "handoff session limit reached");
                    continue;
                }
                let entry = pending.entry(handoff_id).or_insert_with(|| PendingSession {
                    created_at: tokio::time::Instant::now(),
                    source: None,
                    destination: None,
                });
                if entry.source.is_some() {
                    reject_source(source, "duplicate source participant");
                    continue;
                }
                entry.source = Some(source);
                maybe_start_session(
                    handoff_id,
                    &mut pending,
                    &mut active,
                    session_timeout,
                    shutdown.clone(),
                    done_tx.clone(),
                    &sessions,
                    &mut retired,
                    &mut retired_order,
                    max_sessions,
                    &rejection_permits,
                );
            }
            Some(destination) = incoming_rx.recv() => {
                let handoff_id = destination.identity.handoff_id;
                prune_retired(&mut retired, &mut retired_order);
                if retired.contains_key(&handoff_id) || active.contains(&handoff_id) {
                    spawn_destination_rejection(
                        &sessions,
                        &rejection_permits,
                        destination,
                        "handoff session is already active or retired",
                    );
                    continue;
                }
                if !pending.contains_key(&handoff_id) && pending.len() + active.len() >= max_sessions {
                    spawn_destination_overload(
                        &sessions,
                        &rejection_permits,
                        destination,
                    );
                    continue;
                }
                let entry = pending.entry(handoff_id).or_insert_with(|| PendingSession {
                    created_at: tokio::time::Instant::now(),
                    source: None,
                    destination: None,
                });
                if entry.destination.is_some() {
                    spawn_destination_rejection(
                        &sessions,
                        &rejection_permits,
                        destination,
                        "duplicate destination participant",
                    );
                    continue;
                }
                entry.destination = Some(destination);
                maybe_start_session(
                    handoff_id,
                    &mut pending,
                    &mut active,
                    session_timeout,
                    shutdown.clone(),
                    done_tx.clone(),
                    &sessions,
                    &mut retired,
                    &mut retired_order,
                    max_sessions,
                    &rejection_permits,
                );
            }
            _ = cleanup.tick() => {
                prune_retired(&mut retired, &mut retired_order);
                let canceled = pending
                    .iter()
                    .filter_map(|(handoff_id, session)| {
                        session
                            .source
                            .as_ref()
                            .is_some_and(|source| source.cancel.is_cancelled())
                            .then_some(*handoff_id)
                    })
                    .collect::<Vec<_>>();
                for handoff_id in canceled {
                    if let Some(mut session) = pending.remove(&handoff_id) {
                        if let Some(source) = session.source.take() {
                            reject_source(source, "source request was canceled before rendezvous");
                        }
                        if let Some(destination) = session.destination.take() {
                            spawn_destination_rejection(
                                &sessions,
                                &rejection_permits,
                                destination,
                                "source request was canceled before rendezvous",
                            );
                        }
                        retire_session(
                            handoff_id,
                            session_timeout,
                            max_sessions,
                            &mut retired,
                            &mut retired_order,
                        );
                    }
                }
                let disconnected = pending
                    .iter()
                    .filter_map(|(handoff_id, session)| {
                        session
                            .destination
                            .as_ref()
                            .is_some_and(|destination| {
                                destination.connection.peer_closed_now().unwrap_or(true)
                            })
                            .then_some(*handoff_id)
                    })
                    .collect::<Vec<_>>();
                for handoff_id in disconnected {
                    if let Some(mut session) = pending.remove(&handoff_id) {
                        if let Some(source) = session.source.take() {
                            reject_source(source, "destination disconnected before rendezvous");
                        }
                        retire_session(
                            handoff_id,
                            session_timeout,
                            max_sessions,
                            &mut retired,
                            &mut retired_order,
                        );
                    }
                }
                let expired = pending
                    .iter()
                    .filter_map(|(handoff_id, session)| {
                        (session.created_at.elapsed() >= rendezvous_timeout)
                            .then_some(*handoff_id)
                    })
                    .collect::<Vec<_>>();
                for handoff_id in expired {
                    if let Some(mut session) = pending.remove(&handoff_id) {
                        if let Some(source) = session.source.take() {
                            reject_source(source, "handoff participant rendezvous timed out");
                        }
                        if let Some(destination) = session.destination.take() {
                            spawn_destination_rejection(
                                &sessions,
                                &rejection_permits,
                                destination,
                                "handoff participant rendezvous timed out",
                            );
                        }
                        retire_session(
                            handoff_id,
                            session_timeout,
                            max_sessions,
                            &mut retired,
                            &mut retired_order,
                        );
                    }
                }
            }
            else => break,
        }
        #[cfg(test)]
        {
            let pending_sources = pending
                .iter()
                .filter_map(|(handoff_id, session)| session.source.is_some().then_some(*handoff_id))
                .collect();
            let pending_destinations = pending
                .iter()
                .filter_map(|(handoff_id, session)| {
                    session.destination.is_some().then_some(*handoff_id)
                })
                .collect();
            let _ = state_tx.send(SourceHandoffManagerTestState {
                pending_sources,
                pending_destinations,
                active: active.clone(),
                retired: retired.keys().copied().collect(),
            });
        }
    }

    for (_, mut session) in pending {
        if let Some(source) = session.source.take() {
            reject_source(source, "handoff manager is shutting down");
        }
        if let Some(destination) = session.destination.take() {
            spawn_destination_rejection(
                &sessions,
                &rejection_permits,
                destination,
                "handoff manager is shutting down",
            );
        }
    }
    sessions.close();
    sessions.wait().await;
}

#[allow(clippy::too_many_arguments)]
fn maybe_start_session(
    handoff_id: HandoffId,
    pending: &mut HashMap<HandoffId, PendingSession>,
    active: &mut HashSet<HandoffId>,
    session_timeout: Duration,
    shutdown: CancellationToken,
    done_tx: mpsc::Sender<HandoffId>,
    sessions: &TaskTracker,
    retired: &mut HashMap<HandoffId, Instant>,
    retired_order: &mut VecDeque<HandoffId>,
    max_sessions: usize,
    rejection_permits: &Arc<tokio::sync::Semaphore>,
) {
    let ready = pending
        .get(&handoff_id)
        .is_some_and(|session| session.source.is_some() && session.destination.is_some());
    if !ready {
        return;
    }
    let mut session = pending
        .remove(&handoff_id)
        .expect("ready handoff session must remain pending");
    let source = session.source.take().unwrap();
    let destination = session.destination.take().unwrap();
    if source.identity != destination.identity
        || source.order != destination.registration.order
        || source.engine_type != destination.registration.engine_type
    {
        reject_source(source, "handoff participant identity or backend mismatch");
        spawn_destination_rejection(
            sessions,
            rejection_permits,
            destination,
            "handoff participant identity or backend mismatch",
        );
        retire_session(
            handoff_id,
            session_timeout,
            max_sessions,
            retired,
            retired_order,
        );
        return;
    }

    active.insert(handoff_id);
    sessions.spawn(async move {
        let _ = run_source_session(source, destination.connection, session_timeout, shutdown).await;
        let _ = done_tx.try_send(handoff_id);
    });
}

fn retire_session(
    handoff_id: HandoffId,
    session_timeout: Duration,
    max_sessions: usize,
    retired: &mut HashMap<HandoffId, Instant>,
    retired_order: &mut VecDeque<HandoffId>,
) {
    retired.insert(handoff_id, Instant::now() + session_timeout);
    retired_order.push_back(handoff_id);
    while retired_order.len() > max_sessions {
        if let Some(expired) = retired_order.pop_front() {
            retired.remove(&expired);
        }
    }
}

fn reject_source(source: SourceRegistration, message: &str) {
    let _ = source.completion_tx.send(Err(message.to_string()));
}

async fn reject_destination(destination: &mut IncomingBootstrapConnection, message: &str) {
    let _ = destination
        .connection
        .send(BootstrapMessage::ProtocolError {
            message: message.to_string(),
        })
        .await;
}

async fn reject_destination_overloaded(destination: &mut IncomingBootstrapConnection) {
    let _ = destination
        .connection
        .send(BootstrapMessage::Overloaded)
        .await;
}

fn spawn_destination_rejection(
    tasks: &TaskTracker,
    permits: &Arc<tokio::sync::Semaphore>,
    mut destination: IncomingBootstrapConnection,
    message: &'static str,
) {
    let Ok(permit) = permits.clone().try_acquire_owned() else {
        return;
    };
    tasks.spawn(async move {
        let _permit = permit;
        reject_destination(&mut destination, message).await;
    });
}

fn spawn_destination_overload(
    tasks: &TaskTracker,
    permits: &Arc<tokio::sync::Semaphore>,
    mut destination: IncomingBootstrapConnection,
) {
    let Ok(permit) = permits.clone().try_acquire_owned() else {
        return;
    };
    tasks.spawn(async move {
        let _permit = permit;
        reject_destination_overloaded(&mut destination).await;
    });
}

fn prune_retired(
    retired: &mut HashMap<HandoffId, Instant>,
    retired_order: &mut VecDeque<HandoffId>,
) {
    let now = Instant::now();
    while retired_order
        .front()
        .is_some_and(|handoff_id| retired.get(handoff_id).is_none_or(|expiry| *expiry <= now))
    {
        if let Some(handoff_id) = retired_order.pop_front() {
            retired.remove(&handoff_id);
        }
    }
}

enum SourceSessionEvent {
    Scheduler(SchedulerLifecycleEvent),
    Remote(std::result::Result<Option<BootstrapMessage>, String>),
    LocalOutcome {
        action_id: HandoffActionId,
        outcome: HandoffActionOutcome,
    },
    TransferCompleted,
    Deadline,
    SourceCanceled,
    Shutdown,
    TransportFailed(String),
}

struct OutboundBootstrapMessage {
    message: BootstrapMessage,
    ack: Option<oneshot::Sender<std::result::Result<(), String>>>,
}

async fn send_source_session_event(
    event_tx: &mpsc::Sender<SourceSessionEvent>,
    stop: &CancellationToken,
    event: SourceSessionEvent,
) {
    tokio::select! {
        biased;
        _ = stop.cancelled() => {}
        _ = event_tx.send(event) => {}
    }
}

async fn run_source_transport(
    mut connection: BootstrapConnection,
    mut outbound_rx: mpsc::Receiver<OutboundBootstrapMessage>,
    event_tx: mpsc::Sender<SourceSessionEvent>,
    stop: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            _ = stop.cancelled() => return,
            outbound = outbound_rx.recv() => {
                let Some(OutboundBootstrapMessage { message, ack }) = outbound else {
                    return;
                };
                let result = connection.send(message).await.map_err(|error| error.to_string());
                if let Some(ack) = ack {
                    let _ = ack.send(result.clone());
                }
                if let Err(error) = result {
                    send_source_session_event(
                        &event_tx,
                        &stop,
                        SourceSessionEvent::TransportFailed(error),
                    ).await;
                    return;
                }
            }
            message = connection.recv() => {
                let message = message.map_err(|error| error.to_string());
                let closed = matches!(message, Ok(None));
                send_source_session_event(
                    &event_tx,
                    &stop,
                    SourceSessionEvent::Remote(message),
                ).await;
                if closed {
                    return;
                }
            }
        }
    }
}

fn queue_source_message(
    outbound_tx: &mpsc::Sender<OutboundBootstrapMessage>,
    message: BootstrapMessage,
) -> Result<()> {
    outbound_tx
        .try_send(OutboundBootstrapMessage { message, ack: None })
        .map_err(|error| anyhow!("source bootstrap outbound queue failed: {error}"))
}

fn observe_handoff(
    observer: &Option<mpsc::UnboundedSender<NormalizedHandoffEvent>>,
    event: NormalizedHandoffEvent,
) {
    if let Some(observer) = observer {
        let _ = observer.send(event);
    }
}

fn apply_source_held(
    coordinator: &mut HandoffCoordinatorCore,
    actions: &mut VecDeque<IssuedHandoffAction>,
    outbound_tx: &mpsc::Sender<OutboundBootstrapMessage>,
    observer: &Option<mpsc::UnboundedSender<NormalizedHandoffEvent>>,
    observed_source_held: &mut bool,
    handoff_id: HandoffId,
    transfer_timing: HandoffTransferTiming,
) -> Result<()> {
    let next = coordinator.on_fact(HandoffFact::SourceHeld {
        handoff_id,
        transfer_timing,
    })?;
    queue_source_message(
        outbound_tx,
        BootstrapMessage::Fact(HandoffFact::SourceHeld {
            handoff_id,
            transfer_timing,
        }),
    )?;
    actions.extend(next);
    if !*observed_source_held {
        observe_handoff(observer, NormalizedHandoffEvent::SourceHeld);
        *observed_source_held = true;
    }
    Ok(())
}

async fn send_source_message_and_wait(
    outbound_tx: &mpsc::Sender<OutboundBootstrapMessage>,
    message: BootstrapMessage,
    deadline: tokio::time::Instant,
) -> Result<()> {
    let (ack_tx, ack_rx) = oneshot::channel();
    outbound_tx
        .send(OutboundBootstrapMessage {
            message,
            ack: Some(ack_tx),
        })
        .await
        .map_err(|_| anyhow!("source bootstrap transport closed"))?;
    tokio::time::timeout_at(deadline, ack_rx)
        .await
        .map_err(|_| anyhow!("source bootstrap send timed out"))?
        .map_err(|_| anyhow!("source bootstrap send acknowledgement dropped"))?
        .map_err(anyhow::Error::msg)
}

async fn run_source_session(
    source: SourceRegistration,
    connection: BootstrapConnection,
    session_timeout: Duration,
    shutdown: CancellationToken,
) -> Result<()> {
    let SourceRegistration {
        identity,
        order,
        engine_type: _,
        request,
        command_tx,
        mut lifecycle,
        completion_tx,
        cancel,
        observer,
        _permit,
    } = source;
    let _permit = _permit;
    let handoff_id = identity.handoff_id;
    let tasks = TaskTracker::new();
    let stop = CancellationToken::new();
    let action_tasks = TaskTracker::new();
    let action_stop = CancellationToken::new();
    let (event_tx, mut event_rx) = mpsc::channel(SESSION_INBOX_CAPACITY);
    let (outbound_tx, outbound_rx) = mpsc::channel(SESSION_INBOX_CAPACITY);
    let session_started = tokio::time::Instant::now();
    let mut session_deadline = session_started + session_timeout;
    let mut completion_tx = Some(completion_tx);
    let mut observed_destination_accepted = false;
    let mut observed_source_held = false;
    let mut observed_destination_reserved = false;
    let mut pending_submit_action = None;
    let mut pending_source_held = None;
    let mut source_transfer_timing = None;
    let mut destination_transferable_prompt_tokens = None;

    tasks.spawn(run_source_transport(
        connection,
        outbound_rx,
        event_tx.clone(),
        stop.clone(),
    ));

    queue_source_message(&outbound_tx, BootstrapMessage::Registered)?;
    let mut coordinator = HandoffCoordinatorCore::new(handoff_id, order);
    let mut actions = VecDeque::from(coordinator.start()?);
    let mut request = Some(request);
    let deadline = tokio::time::sleep_until(session_deadline);
    tokio::pin!(deadline);

    let result: Result<()> = async {
        loop {
            while let Some(action) = actions.pop_front() {
                match action.action {
                    HandoffAction::SubmitPrefill { handoff_id } => {
                        let Some(request) = request.take() else {
                            bail!("source request was submitted more than once");
                        };
                        if pending_submit_action.replace(action.id).is_some() {
                            bail!("source prefill submission was already pending");
                        }
                        let command_tx = command_tx.clone();
                        let event_tx = event_tx.clone();
                        let cancel = cancel.clone();
                        let shutdown = shutdown.clone();
                        let action_stop = action_stop.clone();
                        let deadline = session_deadline;
                        action_tasks.spawn(async move {
                            let result = tokio::select! {
                                biased;
                                _ = action_stop.cancelled() => return,
                                result = send_source_command(
                                    &command_tx,
                                    SchedulerCommand::SubmitHandoffPrefill {
                                        handoff_id,
                                        request,
                                    },
                                    deadline,
                                    &cancel,
                                    &shutdown,
                                ) => result,
                            };
                            send_source_session_event(
                                &event_tx,
                                &action_stop,
                                SourceSessionEvent::LocalOutcome {
                                    action_id: action.id,
                                    outcome: command_outcome(
                                        result,
                                        SchedulerCommandResultKind::Submitted,
                                    ),
                                },
                            )
                            .await;
                        });
                    }
                    HandoffAction::ReserveDestination { .. }
                    | HandoffAction::ActivateDestination { .. }
                    | HandoffAction::CancelDestination { .. } => {
                        queue_source_message(&outbound_tx, BootstrapMessage::Action(action))?;
                    }
                    HandoffAction::StartTransfer { delay_ms, .. } => {
                        actions.extend(
                            coordinator
                                .on_action_outcome(action.id, HandoffActionOutcome::Scheduled)?,
                        );
                        let event_tx = event_tx.clone();
                        let action_stop = action_stop.clone();
                        action_tasks.spawn(async move {
                            tokio::select! {
                                biased;
                                _ = action_stop.cancelled() => {}
                                _ = tokio::time::sleep(Duration::from_secs_f64(
                                    delay_ms.max(0.0) / 1000.0,
                                )) => {
                                    send_source_session_event(
                                        &event_tx,
                                        &action_stop,
                                        SourceSessionEvent::TransferCompleted,
                                    ).await;
                                }
                            }
                        });
                    }
                    HandoffAction::ReleaseSource { handoff_id }
                    | HandoffAction::CancelSource { handoff_id } => {
                        if matches!(action.action, HandoffAction::ReleaseSource { .. }) {
                            observe_handoff(
                                &observer,
                                NormalizedHandoffEvent::DestinationActivated,
                            );
                        }
                        let (command, expected) = match action.action {
                            HandoffAction::ReleaseSource { .. } => (
                                SchedulerCommand::ReleaseSource { handoff_id },
                                SchedulerCommandResultKind::AppliedOrNoop,
                            ),
                            HandoffAction::CancelSource { .. } => (
                                SchedulerCommand::CancelSource { handoff_id },
                                SchedulerCommandResultKind::AppliedOrNoop,
                            ),
                            _ => unreachable!(),
                        };
                        let command_tx = command_tx.clone();
                        let event_tx = event_tx.clone();
                        let cancel = cancel.clone();
                        let shutdown = shutdown.clone();
                        let action_stop = action_stop.clone();
                        let deadline = session_deadline;
                        action_tasks.spawn(async move {
                            let result = tokio::select! {
                                biased;
                                _ = action_stop.cancelled() => return,
                                result = send_source_command(
                                    &command_tx,
                                    command,
                                    deadline,
                                    &cancel,
                                    &shutdown,
                                ) => result,
                            };
                            send_source_session_event(
                                &event_tx,
                                &action_stop,
                                SourceSessionEvent::LocalOutcome {
                                    action_id: action.id,
                                    outcome: command_outcome(result, expected),
                                },
                            )
                            .await;
                        });
                    }
                    HandoffAction::Complete { .. } => {
                        let completion = coordinator
                            .completion()
                            .expect("complete action requires a terminal outcome");
                        let message = match completion {
                            dynamo_mocker::common::handoff::HandoffCompletion::Success => {
                                BootstrapMessage::Complete
                            }
                            dynamo_mocker::common::handoff::HandoffCompletion::Canceled => {
                                BootstrapMessage::Abort {
                                    message: "handoff canceled".to_string(),
                                }
                            }
                        };
                        if matches!(
                            completion,
                            dynamo_mocker::common::handoff::HandoffCompletion::Success
                        ) {
                            observe_handoff(&observer, NormalizedHandoffEvent::SourceReleased);
                        }
                        send_source_message_and_wait(&outbound_tx, message, session_deadline)
                            .await?;
                        if matches!(
                            completion,
                            dynamo_mocker::common::handoff::HandoffCompletion::Success
                        ) {
                            observe_handoff(&observer, NormalizedHandoffEvent::Completed);
                        }
                        if let Some(completion_tx) = completion_tx.take() {
                            let completion = match completion {
                                dynamo_mocker::common::handoff::HandoffCompletion::Success => {
                                    Ok(())
                                }
                                dynamo_mocker::common::handoff::HandoffCompletion::Canceled => {
                                    Err("handoff canceled".to_string())
                                }
                            };
                            let _ = completion_tx.send(completion);
                        }
                        return Ok(());
                    }
                }
            }

            let event = tokio::select! {
                biased;
                _ = shutdown.cancelled() => SourceSessionEvent::Shutdown,
                _ = cancel.cancelled() => SourceSessionEvent::SourceCanceled,
                _ = &mut deadline => SourceSessionEvent::Deadline,
                event = lifecycle.recv() => {
                    SourceSessionEvent::Scheduler(event.ok_or_else(|| {
                        anyhow!("source lifecycle channel closed")
                    })?)
                }
                event = event_rx.recv() => event.ok_or_else(|| {
                    anyhow!("source handoff session inbox closed")
                })?,
            };
            match event {
                SourceSessionEvent::Scheduler(SchedulerLifecycleEvent::SourceHeld {
                    handoff_id: observed,
                    transfer_timing,
                    ..
                }) if observed == handoff_id => {
                    validate_transfer_timing(transfer_timing)?;
                    source_transfer_timing = Some(transfer_timing);
                    if let Some(Some(timeout_delay)) = transfer_timeout_delay(
                        transfer_timing,
                        destination_transferable_prompt_tokens,
                    ) {
                        session_deadline = session_deadline_with_transfer(
                            session_started,
                            session_timeout,
                            Some(timeout_delay),
                        )?;
                        deadline.as_mut().reset(session_deadline);
                    }
                    if pending_submit_action.is_some() {
                        if let Some(previous) = pending_source_held
                            && previous != transfer_timing
                        {
                            bail!(
                                "source changed its modeled transfer timing before submission ACK"
                            );
                        }
                        pending_source_held = Some(transfer_timing);
                        continue;
                    }
                    apply_source_held(
                        &mut coordinator,
                        &mut actions,
                        &outbound_tx,
                        &observer,
                        &mut observed_source_held,
                        handoff_id,
                        transfer_timing,
                    )?;
                }
                SourceSessionEvent::Scheduler(SchedulerLifecycleEvent::SourceHeld { .. }) => {
                    bail!("source lifecycle event belongs to another handoff")
                }
                SourceSessionEvent::Scheduler(SchedulerLifecycleEvent::DestinationReserved {
                    ..
                }) => bail!("source scheduler emitted a destination lifecycle event"),
                SourceSessionEvent::Remote(Ok(Some(BootstrapMessage::ActionAck {
                    action_id,
                    outcome,
                }))) => {
                    if outcome == HandoffActionOutcome::Accepted && !observed_destination_accepted {
                        observe_handoff(&observer, NormalizedHandoffEvent::DestinationAccepted);
                        observed_destination_accepted = true;
                    }
                    actions.extend(coordinator.on_action_outcome(action_id, outcome)?);
                }
                SourceSessionEvent::Remote(Ok(Some(BootstrapMessage::Fact(
                    HandoffFact::DestinationReserved {
                        handoff_id: observed,
                        transferable_prompt_tokens,
                    },
                )))) if observed == handoff_id => {
                    destination_transferable_prompt_tokens = Some(transferable_prompt_tokens);
                    if !observed_destination_reserved {
                        observe_handoff(&observer, NormalizedHandoffEvent::DestinationReserved);
                        observed_destination_reserved = true;
                    }
                    if let Some(Some(timeout_delay)) = source_transfer_timing
                        .filter(|timing| timing.mode == KvTransferTimingMode::DestinationMissing)
                        .and_then(|timing| {
                            transfer_timeout_delay(timing, Some(transferable_prompt_tokens))
                        })
                    {
                        session_deadline = session_deadline_with_transfer(
                            session_started,
                            session_timeout,
                            Some(timeout_delay),
                        )?;
                        deadline.as_mut().reset(session_deadline);
                    }
                    actions.extend(coordinator.on_fact(HandoffFact::DestinationReserved {
                        handoff_id,
                        transferable_prompt_tokens,
                    })?);
                }
                SourceSessionEvent::Remote(Ok(Some(BootstrapMessage::Abort { message })))
                | SourceSessionEvent::Remote(Ok(Some(BootstrapMessage::ProtocolError {
                    message,
                }))) => bail!("{message}"),
                SourceSessionEvent::Remote(Ok(Some(BootstrapMessage::Overloaded))) => {
                    bail!("destination overloaded")
                }
                SourceSessionEvent::Remote(Ok(Some(other))) => {
                    bail!("unexpected destination bootstrap message: {other:?}")
                }
                SourceSessionEvent::Remote(Ok(None)) => bail!("destination disconnected"),
                SourceSessionEvent::Remote(Err(error))
                | SourceSessionEvent::TransportFailed(error) => bail!("{error}"),
                SourceSessionEvent::LocalOutcome { action_id, outcome } => {
                    let completes_submit = pending_submit_action == Some(action_id);
                    let submitted = matches!(&outcome, HandoffActionOutcome::Submitted);
                    actions.extend(coordinator.on_action_outcome(action_id, outcome)?);
                    if completes_submit {
                        pending_submit_action = None;
                        if let Some(transfer_timing) = pending_source_held.take() {
                            if !submitted {
                                bail!("source was held after its submission failed");
                            }
                            apply_source_held(
                                &mut coordinator,
                                &mut actions,
                                &outbound_tx,
                                &observer,
                                &mut observed_source_held,
                                handoff_id,
                                transfer_timing,
                            )?;
                        }
                    }
                }
                SourceSessionEvent::TransferCompleted => {
                    actions.extend(
                        coordinator.on_fact(HandoffFact::TransferCompleted { handoff_id })?,
                    );
                }
                SourceSessionEvent::Deadline => bail!("handoff session timed out"),
                SourceSessionEvent::SourceCanceled => bail!("source request was canceled"),
                SourceSessionEvent::Shutdown => bail!("mocker is shutting down"),
            }
        }
    }
    .await;

    action_stop.cancel();
    action_tasks.close();
    action_tasks.wait().await;

    if let Err(error) = &result {
        let cleanup_deadline =
            tokio::time::Instant::now() + session_timeout.max(PARTICIPANT_RENDEZVOUS_TIMEOUT);
        let cleanup = send_cleanup_command(
            &command_tx,
            SchedulerCommand::CancelSource { handoff_id },
            cleanup_deadline,
        );
        let abort = send_source_message_and_wait(
            &outbound_tx,
            BootstrapMessage::Abort {
                message: error.to_string(),
            },
            cleanup_deadline,
        );
        let _ = tokio::join!(cleanup, abort);
        if let Some(completion_tx) = completion_tx.take() {
            let _ = completion_tx.send(Err(error.to_string()));
        }
    }

    stop.cancel();
    tasks.close();
    tasks.wait().await;
    result
}

pub(crate) async fn run_destination_session(
    mut connection: BootstrapConnection,
    request: DirectRequest,
    command_tx: mpsc::Sender<SchedulerCommandEnvelope>,
    mut lifecycle: HandoffEventRoute,
    cancel: CancellationToken,
    session_timeout: Duration,
    shutdown: CancellationToken,
) -> Result<()> {
    let handoff_id = connection.identity().handoff_id;
    let mut request = Some(request);
    let mut outcomes = HashMap::<HandoffActionId, (HandoffAction, HandoffActionOutcome)>::new();
    let mut complete = false;
    let mut session_started = None;
    let mut session_deadline = tokio::time::Instant::now() + PARTICIPANT_RENDEZVOUS_TIMEOUT;
    let mut transfer_timing: Option<HandoffTransferTiming> = None;
    let mut transferable_prompt_tokens = None;
    let action_stop = CancellationToken::new();
    let action_tasks = TaskTracker::new();
    let (action_result_tx, mut action_result_rx) =
        mpsc::channel::<(IssuedHandoffAction, HandoffActionOutcome)>(1);
    let mut pending_action: Option<IssuedHandoffAction> = None;
    let mut reserve_ack_sent = false;
    let mut pending_destination_reserved = None;
    let mut destination_reserved_sent = false;
    let deadline = tokio::time::sleep_until(session_deadline);
    tokio::pin!(deadline);

    let result: Result<bool> = async {
        loop {
            tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            _ = cancel.cancelled() => break,
            _ = &mut deadline => break,
            result = action_result_rx.recv(), if pending_action.is_some() => {
                let Some((action, outcome)) = result else {
                    bail!("destination action executor closed");
                };
                let expected = pending_action
                    .take()
                    .expect("destination action result requires pending ownership");
                if expected != action {
                    bail!("destination action executor returned a mismatched action");
                }
                if outcomes.len() >= 16 {
                    bail!("bootstrap action outcome cache exceeded its bound");
                }
                outcomes.insert(action.id, (action.action, outcome.clone()));
                let accepted_reservation = matches!(
                    (action.action, &outcome),
                    (
                        HandoffAction::ReserveDestination { .. },
                        HandoffActionOutcome::Accepted
                    )
                );
                connection.send(BootstrapMessage::ActionAck {
                    action_id: action.id,
                    outcome,
                }).await?;
                if accepted_reservation {
                    reserve_ack_sent = true;
                    if let Some(transferable_prompt_tokens) = pending_destination_reserved.take() {
                        connection
                            .send(BootstrapMessage::Fact(HandoffFact::DestinationReserved {
                                handoff_id,
                                transferable_prompt_tokens,
                            }))
                            .await?;
                        destination_reserved_sent = true;
                    }
                } else if pending_destination_reserved.is_some()
                    && matches!(action.action, HandoffAction::ReserveDestination { .. })
                {
                    bail!("destination reservation lifecycle accompanied a failed reservation");
                }
            }
            event = lifecycle.recv() => {
                let Some(event) = event else {
                    break;
                };
                match event {
                    SchedulerLifecycleEvent::DestinationReserved {
                        handoff_id: observed,
                        transferable_prompt_tokens: observed_transferable_prompt_tokens,
                        ..
                    }
                        if observed == handoff_id =>
                    {
                        if destination_reserved_sent || pending_destination_reserved.is_some() {
                            continue;
                        }
                        transferable_prompt_tokens = Some(observed_transferable_prompt_tokens);
                        if let Some(Some(timeout_delay)) = transfer_timing
                            .filter(|timing| {
                                timing.mode == KvTransferTimingMode::DestinationMissing
                            })
                            .and_then(|timing| {
                                transfer_timeout_delay(
                                    timing,
                                    Some(observed_transferable_prompt_tokens),
                                )
                            })
                        {
                            session_deadline = session_deadline_with_transfer(
                                session_started.ok_or_else(|| {
                                    anyhow!("destination reserved before source registration")
                                })?,
                                session_timeout,
                                Some(timeout_delay),
                            )?;
                            deadline.as_mut().reset(session_deadline);
                        }
                        if reserve_ack_sent {
                            connection
                                .send(BootstrapMessage::Fact(HandoffFact::DestinationReserved {
                                    handoff_id,
                                    transferable_prompt_tokens:
                                        observed_transferable_prompt_tokens,
                                }))
                                .await?;
                            destination_reserved_sent = true;
                        } else if matches!(
                            pending_action.map(|action| action.action),
                            Some(HandoffAction::ReserveDestination { .. })
                        ) {
                            pending_destination_reserved =
                                Some(observed_transferable_prompt_tokens);
                        } else {
                            bail!("destination reserved before its action was pending");
                        }
                    }
                    SchedulerLifecycleEvent::SourceHeld { .. } => {
                        bail!("destination scheduler emitted a source lifecycle event")
                    }
                    _ => bail!("destination lifecycle event belongs to another handoff"),
                }
            }
            message = connection.recv() => {
                let Some(message) = message? else {
                    break;
                };
                match message {
                    BootstrapMessage::Registered => {
                        if session_started.is_some() {
                            bail!("source registered the destination more than once");
                        }
                        let started = tokio::time::Instant::now();
                        session_started = Some(started);
                        session_deadline = started + session_timeout;
                        deadline.as_mut().reset(session_deadline);
                    }
                    BootstrapMessage::Fact(HandoffFact::SourceHeld {
                        handoff_id: observed,
                        transfer_timing: observed_timing,
                    }) if observed == handoff_id => {
                        validate_transfer_timing(observed_timing)?;
                        if let Some(previous) = transfer_timing
                            && previous != observed_timing
                        {
                            bail!("source changed the modeled transfer timing");
                        }
                        transfer_timing = Some(observed_timing);
                        if let Some(Some(timeout_delay)) = transfer_timeout_delay(
                            observed_timing,
                            transferable_prompt_tokens,
                        ) {
                            session_deadline = session_deadline_with_transfer(
                                session_started.ok_or_else(|| {
                                    anyhow!("source fact arrived before registration")
                                })?,
                                session_timeout,
                                Some(timeout_delay),
                            )?;
                            deadline.as_mut().reset(session_deadline);
                        }
                    }
                    BootstrapMessage::Action(action) => {
                        if let Some((previous_action, outcome)) = outcomes.get(&action.id) {
                            if previous_action != &action.action {
                                bail!("bootstrap action ID was reused with different content");
                            }
                            connection.send(BootstrapMessage::ActionAck {
                                action_id: action.id,
                                outcome: outcome.clone(),
                            }).await?;
                            continue;
                        }
                        if let Some(pending) = pending_action {
                            if pending != action {
                                bail!("destination received a new action while another was pending");
                            }
                            continue;
                        }
                        if outcomes.len() >= 16 {
                            bail!("bootstrap action outcome cache exceeded its bound");
                        }
                        let (command, expected) = match prepare_destination_action(
                            &mut request,
                            action,
                        ) {
                            Ok(prepared) => prepared,
                            Err(outcome) => {
                                outcomes.insert(action.id, (action.action, outcome.clone()));
                                connection.send(BootstrapMessage::ActionAck {
                                    action_id: action.id,
                                    outcome,
                                }).await?;
                                continue;
                            }
                        };
                        pending_action = Some(action);
                        let command_tx = command_tx.clone();
                        let cancel = cancel.clone();
                        let shutdown = shutdown.clone();
                        let action_stop = action_stop.clone();
                        let result_tx = action_result_tx.clone();
                        let action_deadline = session_deadline;
                        action_tasks.spawn(async move {
                            let result = tokio::select! {
                                biased;
                                _ = action_stop.cancelled() => return,
                                result = send_destination_command(
                                    &command_tx,
                                    command,
                                    action_deadline,
                                    &cancel,
                                    &shutdown,
                                ) => result,
                            };
                            let outcome = command_outcome(result, expected);
                            tokio::select! {
                                biased;
                                _ = action_stop.cancelled() => {}
                                _ = result_tx.send((action, outcome)) => {}
                            }
                        });
                    }
                    BootstrapMessage::Complete => {
                        let activation_applied = outcomes.values().any(|(action, outcome)| {
                            matches!(action, HandoffAction::ActivateDestination { .. })
                                && matches!(outcome, HandoffActionOutcome::Applied)
                        });
                        if pending_action.is_some() || !activation_applied {
                            bail!("source completed before destination activation finished");
                        }
                        complete = true;
                        break;
                    }
                    BootstrapMessage::Abort { .. } | BootstrapMessage::ProtocolError { .. } => break,
                    other => bail!("unexpected source bootstrap message: {other:?}"),
                }
            }
            }
        }
        Ok(complete)
    }
    .await;

    action_stop.cancel();
    action_tasks.close();
    action_tasks.wait().await;

    if !matches!(result, Ok(true)) {
        let cleanup_deadline =
            tokio::time::Instant::now() + session_timeout.max(PARTICIPANT_RENDEZVOUS_TIMEOUT);
        let _ = send_cleanup_command(
            &command_tx,
            SchedulerCommand::CancelDestination { handoff_id },
            cleanup_deadline,
        )
        .await;
    }
    match result {
        Ok(true) => Ok(()),
        Ok(false) => bail!("destination handoff ended before completion"),
        Err(error) => Err(error),
    }
}

fn prepare_destination_action(
    request: &mut Option<DirectRequest>,
    action: IssuedHandoffAction,
) -> std::result::Result<(SchedulerCommand, SchedulerCommandResultKind), HandoffActionOutcome> {
    let (command, expected) = match action.action {
        HandoffAction::ReserveDestination { handoff_id } => {
            let Some(request) = request.take() else {
                return Err(HandoffActionOutcome::Failed(
                    "destination request was reserved more than once".to_string(),
                ));
            };
            (
                SchedulerCommand::ReserveDestination {
                    handoff_id,
                    request,
                },
                SchedulerCommandResultKind::Accepted,
            )
        }
        HandoffAction::ActivateDestination { handoff_id } => (
            SchedulerCommand::ActivateDestination { handoff_id },
            SchedulerCommandResultKind::Applied,
        ),
        HandoffAction::CancelDestination { handoff_id } => (
            SchedulerCommand::CancelDestination { handoff_id },
            SchedulerCommandResultKind::AppliedOrNoop,
        ),
        _ => {
            return Err(HandoffActionOutcome::Failed(
                "source sent a non-destination scheduler action".to_string(),
            ));
        }
    };
    Ok((command, expected))
}

async fn send_source_command(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    command: SchedulerCommand,
    deadline: tokio::time::Instant,
    cancel: &CancellationToken,
    shutdown: &CancellationToken,
) -> Result<SchedulerCommandEffects> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => bail!("mocker is shutting down"),
        _ = cancel.cancelled() => bail!("source request was canceled"),
        _ = tokio::time::sleep_until(deadline) => bail!("handoff session timed out"),
        result = send_command(command_tx, command) => result,
    }
}

async fn send_destination_command(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    command: SchedulerCommand,
    deadline: tokio::time::Instant,
    cancel: &CancellationToken,
    shutdown: &CancellationToken,
) -> Result<SchedulerCommandEffects> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => bail!("mocker is shutting down"),
        _ = cancel.cancelled() => bail!("destination request was canceled"),
        _ = tokio::time::sleep_until(deadline) => bail!("handoff session timed out"),
        result = send_command(command_tx, command) => result,
    }
}

async fn send_cleanup_command(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    command: SchedulerCommand,
    deadline: tokio::time::Instant,
) -> Result<SchedulerCommandEffects> {
    tokio::time::timeout_at(deadline, send_command(command_tx, command))
        .await
        .map_err(|_| anyhow!("scheduler cleanup command timed out"))?
}

async fn send_command(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    command: SchedulerCommand,
) -> Result<SchedulerCommandEffects> {
    let (reply_tx, reply_rx) = oneshot::channel();
    command_tx
        .send(SchedulerCommandEnvelope {
            command,
            reply: reply_tx,
        })
        .await
        .map_err(|_| anyhow!("scheduler command channel closed"))?;
    reply_rx
        .await
        .map_err(|_| anyhow!("scheduler command reply was dropped"))?
}

pub(crate) async fn cancel_destination(
    command_tx: &mpsc::Sender<SchedulerCommandEnvelope>,
    handoff_id: HandoffId,
    timeout: Duration,
) {
    let _ = send_cleanup_command(
        command_tx,
        SchedulerCommand::CancelDestination { handoff_id },
        tokio::time::Instant::now() + timeout,
    )
    .await;
}

#[derive(Clone, Copy)]
enum SchedulerCommandResultKind {
    Submitted,
    Accepted,
    Applied,
    AppliedOrNoop,
}

fn command_outcome(
    result: Result<SchedulerCommandEffects>,
    expected: SchedulerCommandResultKind,
) -> HandoffActionOutcome {
    let effects = match result {
        Ok(effects) => effects,
        Err(error) => return HandoffActionOutcome::Failed(error.to_string()),
    };
    match (expected, effects.result) {
        (SchedulerCommandResultKind::Submitted, SchedulerCommandResult::Submitted(_)) => {
            HandoffActionOutcome::Submitted
        }
        (
            SchedulerCommandResultKind::Accepted,
            SchedulerCommandResult::DestinationAccepted { .. },
        ) => HandoffActionOutcome::Accepted,
        (SchedulerCommandResultKind::Applied, SchedulerCommandResult::Applied) => {
            HandoffActionOutcome::Applied
        }
        (SchedulerCommandResultKind::AppliedOrNoop, SchedulerCommandResult::Applied) => {
            HandoffActionOutcome::Applied
        }
        (SchedulerCommandResultKind::AppliedOrNoop, SchedulerCommandResult::Noop) => {
            HandoffActionOutcome::Noop
        }
        _ => HandoffActionOutcome::Failed("scheduler returned an unexpected result".to_string()),
    }
}

#[cfg(test)]
#[path = "handoff_tests.rs"]
mod tests;
