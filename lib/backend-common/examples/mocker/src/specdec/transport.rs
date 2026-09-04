// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use futures::{Sink, SinkExt, Stream, StreamExt};
use tmq::{Context, Multipart, dealer, router};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::metrics::{QueueDepthGuard, SpecdecMetrics};
use super::protocol::{
    Cleanup, CleanupAck, Complete, DraftIdentity, Envelope, ErrorCode, ErrorPayload, FailureState,
    Heartbeat, HeartbeatAck, Hello, HelloAck, MAX_FRAME_BYTES, Message, Proposal,
    SequenceValidator, Start, StartAck, proposal_digest,
};
use super::queue::{
    FakeScheduler, JobHandle, JobSpec, QueueError, QueueEvent, SchedulerConfig, TokenMode,
};

const TASK_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    Configuration,
    Connect,
    Timeout,
    Closed,
    Backpressure,
    Protocol,
    Identity,
    Queue,
    Task,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub state: FailureState,
    message: &'static str,
}

impl TransportError {
    pub(super) fn new(
        kind: TransportErrorKind,
        state: FailureState,
        message: &'static str,
    ) -> Self {
        Self {
            kind,
            state,
            message,
        }
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for TransportError {}

#[derive(Debug, Clone)]
pub struct DraftServerConfig {
    pub bind_address: String,
    pub transport_hwm: i32,
    pub outbound_capacity: usize,
    pub prefill_duration: Duration,
    pub token_interval: Duration,
    pub token_mode: TokenMode,
    pub scheduler: SchedulerConfig,
}

impl DraftServerConfig {
    fn validate(&self) -> Result<(), TransportError> {
        if self.bind_address.is_empty() || self.bind_address.chars().any(char::is_control) {
            return Err(TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft bind address is invalid",
            ));
        }
        if self.transport_hwm <= 0 || self.outbound_capacity == 0 {
            return Err(TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft transport bounds must be positive",
            ));
        }
        if self.outbound_capacity > Semaphore::MAX_PERMITS {
            return Err(TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft server capacity exceeds the Tokio primitive limit",
            ));
        }
        self.scheduler.validate().map_err(|_| {
            TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft scheduler configuration is invalid",
            )
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DraftClientConfig {
    pub transport_hwm: i32,
    pub outbound_capacity: usize,
    pub session_capacity: usize,
    pub max_sessions: usize,
    pub handshake_timeout: Duration,
    pub start_timeout: Duration,
    pub inactivity_timeout: Duration,
    pub cleanup_timeout: Duration,
}

impl Default for DraftClientConfig {
    fn default() -> Self {
        Self {
            transport_hwm: 64,
            outbound_capacity: 64,
            session_capacity: 16,
            max_sessions: 64,
            handshake_timeout: Duration::from_secs(2),
            start_timeout: Duration::from_secs(2),
            inactivity_timeout: Duration::from_secs(2),
            cleanup_timeout: Duration::from_secs(2),
        }
    }
}

impl DraftClientConfig {
    fn validate(&self) -> Result<(), TransportError> {
        if self.transport_hwm <= 0
            || self.outbound_capacity == 0
            || self.session_capacity == 0
            || self.max_sessions == 0
            || self.handshake_timeout.is_zero()
            || self.start_timeout.is_zero()
            || self.inactivity_timeout.is_zero()
            || self.cleanup_timeout.is_zero()
        {
            return Err(TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft client bounds and timeouts must be positive",
            ));
        }
        if self.outbound_capacity > Semaphore::MAX_PERMITS
            || self.session_capacity > Semaphore::MAX_PERMITS
            || self.max_sessions > Semaphore::MAX_PERMITS
        {
            return Err(TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "draft client capacity exceeds the Tokio primitive limit",
            ));
        }
        Ok(())
    }
}

struct ServerOutbound {
    peer: Vec<u8>,
    envelope: Envelope,
}

struct ServerLifecycle {
    cancel: CancellationToken,
    metrics: Arc<SpecdecMetrics>,
}

struct JobForwardContext {
    peer: Vec<u8>,
    outbound: mpsc::Sender<ServerOutbound>,
    identity: DraftIdentity,
    metrics: Arc<SpecdecMetrics>,
}

struct ServerSession {
    peer: Vec<u8>,
    last_heartbeat: Instant,
    cancel: CancellationToken,
    forwarder: JoinHandle<()>,
}

pub struct DraftServer {
    cancel: CancellationToken,
    scheduler: Arc<FakeScheduler>,
    reader: Mutex<Option<JoinHandle<()>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    metrics: Arc<SpecdecMetrics>,
}

impl DraftServer {
    pub async fn bind(
        config: DraftServerConfig,
        identity: DraftIdentity,
    ) -> Result<Self, TransportError> {
        Self::bind_with_metrics(config, identity, Arc::new(SpecdecMetrics::default())).await
    }

    pub(crate) async fn bind_with_metrics(
        config: DraftServerConfig,
        identity: DraftIdentity,
        metrics: Arc<SpecdecMetrics>,
    ) -> Result<Self, TransportError> {
        config.validate()?;
        identity.validate().map_err(|_| {
            TransportError::new(
                TransportErrorKind::Identity,
                FailureState::NotStarted,
                "draft identity is invalid",
            )
        })?;
        let socket = router(&Context::new())
            .set_linger(0)
            .set_sndhwm(config.transport_hwm)
            .set_rcvhwm(config.transport_hwm)
            .set_maxmsgsize(MAX_FRAME_BYTES as i64)
            .bind(&config.bind_address)
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::Connect,
                    FailureState::NotStarted,
                    "failed to bind draft transport",
                )
            })?;
        // A ROUTER serves every target connection. Non-mandatory routing makes
        // a reply to a disconnected peer an atomic drop instead of exposing
        // tmq's partially-sent multipart buffer to the next healthy peer.
        socket.set_router_mandatory(false).map_err(|_| {
            TransportError::new(
                TransportErrorKind::Configuration,
                FailureState::NotStarted,
                "failed to configure non-mandatory draft routing",
            )
        })?;
        let (sink, stream) = socket.split();
        let scheduler = FakeScheduler::start(config.scheduler).map_err(queue_transport_error)?;
        let cancel = CancellationToken::new();
        let (outbound_tx, outbound_rx) = mpsc::channel(config.outbound_capacity);
        let writer_cancel = cancel.clone();
        let writer_identity = identity.clone();
        let writer = tokio::spawn(run_server_writer(
            sink,
            outbound_rx,
            writer_identity,
            writer_cancel,
        ));
        let reader_scheduler = scheduler.clone();
        let reader = tokio::spawn(run_server_reader(
            stream,
            outbound_tx,
            identity,
            config,
            reader_scheduler,
            ServerLifecycle {
                cancel: cancel.clone(),
                metrics: metrics.clone(),
            },
        ));

        Ok(Self {
            cancel,
            scheduler,
            reader: Mutex::new(Some(reader)),
            writer: Mutex::new(Some(writer)),
            metrics,
        })
    }

    pub fn active_sessions(&self) -> usize {
        self.metrics.snapshot().active_sessions as usize
    }

    #[cfg(test)]
    pub(crate) fn metrics_snapshot(&self) -> super::metrics::SpecdecMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub async fn shutdown(&self) -> Result<(), TransportError> {
        self.cancel.cancel();
        let reader = self.reader.lock().await.take();
        let writer = self.writer.lock().await.take();
        let reader_result = join_task(reader).await;
        let writer_result = join_task(writer).await;
        let scheduler_result = self
            .scheduler
            .shutdown()
            .await
            .map_err(queue_transport_error);
        let result = reader_result.and(writer_result).and(scheduler_result);
        let active_sessions = self.active_sessions();
        if active_sessions == 0 {
            tracing::info!("mock speculative draft transport stopped with zero active sessions");
        } else {
            tracing::warn!(
                active_sessions,
                "mock speculative draft transport stopped with active sessions"
            );
        }
        result
    }
}

impl Drop for DraftServer {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn run_server_writer<S>(
    mut sink: S,
    mut outbound: mpsc::Receiver<ServerOutbound>,
    identity: DraftIdentity,
    cancel: CancellationToken,
) where
    S: Sink<Multipart> + Unpin,
    S::Error: fmt::Display,
{
    while let Some(message) = tokio::select! {
        biased;
        _ = cancel.cancelled() => None,
        message = outbound.recv() => message,
    } {
        let request_id = message.envelope.request_id;
        let is_start_ack = matches!(&message.envelope.message, Message::StartAck(_));
        let frame = match message.envelope.encode() {
            Ok(frame) => frame,
            Err(error) => {
                tracing::error!(%error, "server attempted to send an invalid protocol frame");
                cancel.cancel();
                break;
            }
        };
        let send = sink.send(Multipart::from(vec![message.peer, frame]));
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = send => result,
        };
        if let Err(error) = result {
            tracing::warn!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                %error,
                "draft transport writer stopped"
            );
            cancel.cancel();
            break;
        }
        if is_start_ack {
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                "draft server wrote START_ACK"
            );
        }
    }
}

async fn run_server_reader<S>(
    mut stream: S,
    outbound: mpsc::Sender<ServerOutbound>,
    identity: DraftIdentity,
    config: DraftServerConfig,
    scheduler: Arc<FakeScheduler>,
    lifecycle: ServerLifecycle,
) where
    S: Stream<Item = tmq::Result<Multipart>> + Unpin,
{
    let lease = Duration::from_millis(u64::from(identity.orphan_cleanup_timeout_ms));
    let max_sessions = config
        .scheduler
        .queue_capacity
        .saturating_add(config.scheduler.concurrency);
    let mut sessions = HashMap::<Uuid, ServerSession>::new();
    let mut authenticated_peers = HashMap::<Vec<u8>, Instant>::new();

    loop {
        let reap_deadline = next_reap_deadline(&sessions, &authenticated_peers, lease);
        tokio::select! {
            biased;
            _ = lifecycle.cancel.cancelled() => break,
            _ = tokio::time::sleep_until(reap_deadline) => {
                reap_orphans(&mut sessions, lease, &identity, &lifecycle.metrics).await;
                authenticated_peers.retain(|peer, last_seen| {
                    last_seen.elapsed() < lease
                        || sessions.values().any(|session| session.peer == *peer)
                });
            },
            incoming = stream.next() => {
                let Some(incoming) = incoming else {
                    lifecycle.cancel.cancel();
                    break;
                };
                let frames = match incoming {
                    Ok(frames) => frames,
                    Err(error) => {
                        tracing::warn!(%error, "draft transport reader stopped");
                        lifecycle.cancel.cancel();
                        break;
                    }
                };
                handle_server_frames(
                    frames,
                    &outbound,
                    &identity,
                    &config,
                    &scheduler,
                    &mut sessions,
                    &mut authenticated_peers,
                    max_sessions,
                    &lifecycle.metrics,
                )
                .await;
            }
        }
    }

    for (_, session) in sessions.drain() {
        reap_session(session).await;
    }
    lifecycle.metrics.set_active_sessions(0);
}

fn next_reap_deadline(
    sessions: &HashMap<Uuid, ServerSession>,
    authenticated_peers: &HashMap<Vec<u8>, Instant>,
    lease: Duration,
) -> Instant {
    let session_deadlines = sessions
        .values()
        .map(|session| session.last_heartbeat + lease);
    let idle_peer_deadlines = authenticated_peers
        .iter()
        .filter(|(peer, _)| {
            !sessions
                .values()
                .any(|session| session.peer.as_slice() == peer.as_slice())
        })
        .map(|(_, last_seen)| *last_seen + lease);
    session_deadlines
        .chain(idle_peer_deadlines)
        .min()
        .unwrap_or_else(|| Instant::now() + lease)
}

// These are the independently owned pieces of one reader-loop state. Keeping them explicit makes
// the mutable session-table borrows last for only one decoded frame.
#[allow(clippy::too_many_arguments)]
async fn handle_server_frames(
    mut frames: Multipart,
    outbound: &mpsc::Sender<ServerOutbound>,
    identity: &DraftIdentity,
    config: &DraftServerConfig,
    scheduler: &Arc<FakeScheduler>,
    sessions: &mut HashMap<Uuid, ServerSession>,
    authenticated_peers: &mut HashMap<Vec<u8>, Instant>,
    max_sessions: usize,
    metrics: &Arc<SpecdecMetrics>,
) {
    if frames.len() != 2 {
        tracing::warn!(
            frame_count = frames.len(),
            "rejected malformed ROUTER message"
        );
        return;
    }
    let peer = frames.pop_front().expect("checked frame count").to_vec();
    let body = frames.pop_front().expect("checked frame count");
    let envelope = match Envelope::decode(body.as_ref()) {
        Ok(envelope) => envelope,
        Err(error) => {
            tracing::warn!(%error, "rejected invalid draft protocol frame");
            return;
        }
    };
    let request_id = envelope.request_id;
    let sequence = envelope.sequence;
    let is_start = matches!(&envelope.message, Message::Start(_));
    match envelope.message {
        Message::Hello(hello) if sequence == 0 => {
            let message = if hello.expected == *identity
                && (authenticated_peers.contains_key(&peer)
                    || authenticated_peers.len() < max_sessions)
            {
                authenticated_peers.insert(peer.clone(), Instant::now());
                Message::HelloAck(HelloAck {
                    identity: identity.clone(),
                })
            } else if hello.expected == *identity {
                Message::Error(ErrorPayload::new(
                    ErrorCode::QueueFull,
                    FailureState::NotStarted,
                ))
            } else {
                authenticated_peers.remove(&peer);
                Message::Error(ErrorPayload::new(
                    ErrorCode::IdentityMismatch,
                    FailureState::ProtocolInvalid,
                ))
            };
            let _ = send_server(outbound, peer, Envelope::new(request_id, 0, message)).await;
        }
        Message::Start(start) if sequence == 0 => {
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                "draft server received START"
            );
            let Some(last_seen) = authenticated_peers.get_mut(&peer) else {
                metrics.start_rejected();
                let _ = send_server_error(
                    outbound,
                    peer,
                    request_id,
                    sequence,
                    ErrorCode::IdentityMismatch,
                    FailureState::ProtocolInvalid,
                )
                .await;
                return;
            };
            *last_seen = Instant::now();
            if sessions.contains_key(&request_id) {
                metrics.start_rejected();
                let _ = send_server_error(
                    outbound,
                    peer,
                    request_id,
                    0,
                    ErrorCode::DuplicateRequest,
                    FailureState::Accepted,
                )
                .await;
                return;
            }
            if sessions.len() >= max_sessions {
                metrics.start_rejected();
                metrics.queue_rejected();
                let _ = send_server_error(
                    outbound,
                    peer,
                    request_id,
                    sequence,
                    ErrorCode::QueueFull,
                    FailureState::NotStarted,
                )
                .await;
                return;
            }
            let prompt_token_ids = start.prompt_token_ids;
            let queue_depth = metrics.enter_queue();
            let job = scheduler
                .submit(JobSpec {
                    request_id,
                    prompt_token_ids: prompt_token_ids.clone(),
                    max_output_tokens: start.max_output_tokens,
                    prefill_duration: config.prefill_duration,
                    token_interval: config.token_interval,
                    token_mode: config.token_mode,
                })
                .await;
            let mut job = match job {
                Ok(job) => job,
                Err(error) => {
                    metrics.start_rejected();
                    let code = if error == QueueError::Full {
                        metrics.queue_rejected();
                        ErrorCode::QueueFull
                    } else {
                        ErrorCode::Internal
                    };
                    let _ = send_server_error(
                        outbound,
                        peer,
                        request_id,
                        0,
                        code,
                        FailureState::NotStarted,
                    )
                    .await;
                    return;
                }
            };
            let job_cancel = job.cancellation_token();
            let forward = JobForwardContext {
                peer: peer.clone(),
                outbound: outbound.clone(),
                identity: identity.clone(),
                metrics: metrics.clone(),
            };
            let (start_forwarder, wait_for_start_ack) = oneshot::channel();
            let forwarder = tokio::spawn(async move {
                if wait_for_start_ack.await.is_ok() {
                    forward_job(&mut job, forward, queue_depth).await;
                }
            });
            sessions.insert(
                request_id,
                ServerSession {
                    peer: peer.clone(),
                    last_heartbeat: Instant::now(),
                    cancel: job_cancel,
                    forwarder,
                },
            );
            metrics.set_active_sessions(sessions.len());
            if send_server(
                outbound,
                peer,
                Envelope::new(request_id, 0, Message::StartAck(StartAck::default())),
            )
            .await
            .is_err()
            {
                metrics.start_rejected();
                if let Some(session) = sessions.remove(&request_id) {
                    metrics.set_active_sessions(sessions.len());
                    reap_session(session).await;
                }
                return;
            }
            metrics.start_accepted();
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                "draft server queued START_ACK"
            );
            let _ = start_forwarder.send(());
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                "mock speculative draft accepted request"
            );
        }
        Message::Heartbeat(_) => {
            let Some(last_seen) = authenticated_peers.get_mut(&peer) else {
                let _ = send_server_error(
                    outbound,
                    peer,
                    request_id,
                    sequence,
                    ErrorCode::IdentityMismatch,
                    FailureState::ProtocolInvalid,
                )
                .await;
                return;
            };
            *last_seen = Instant::now();
            for session in sessions
                .values_mut()
                .filter(|session| session.peer == peer && !session.forwarder.is_finished())
            {
                session.last_heartbeat = Instant::now();
            }
            let _ = send_server(
                outbound,
                peer,
                Envelope::new(
                    request_id,
                    sequence,
                    Message::HeartbeatAck(HeartbeatAck::default()),
                ),
            )
            .await;
        }
        Message::Cleanup(_) => {
            if !authenticated_peers.contains_key(&peer)
                || sessions
                    .get(&request_id)
                    .is_none_or(|session| session.peer != peer)
            {
                let _ = send_server_error(
                    outbound,
                    peer,
                    request_id,
                    sequence,
                    ErrorCode::UnknownRequest,
                    FailureState::NotStarted,
                )
                .await;
                return;
            }
            let session = sessions
                .remove(&request_id)
                .expect("session ownership checked");
            metrics.set_active_sessions(sessions.len());
            reap_session(session).await;
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                active_sessions = sessions.len(),
                "mock speculative draft cleaned request"
            );
            if send_server(
                outbound,
                peer,
                Envelope::new(
                    request_id,
                    sequence,
                    Message::CleanupAck(CleanupAck::default()),
                ),
            )
            .await
            .is_ok()
            {
                metrics.cleanup_acknowledged();
            }
        }
        _ => {
            if is_start {
                metrics.start_rejected();
            }
            let state = if sessions.contains_key(&request_id) {
                FailureState::Accepted
            } else {
                FailureState::NotStarted
            };
            let _ = send_server_error(
                outbound,
                peer,
                request_id,
                sequence,
                ErrorCode::InvalidMessage,
                state,
            )
            .await;
        }
    }
}

async fn forward_job(
    job: &mut JobHandle,
    context: JobForwardContext,
    mut queue_depth: QueueDepthGuard,
) {
    let JobForwardContext {
        peer,
        outbound,
        identity,
        metrics,
    } = context;
    let request_id = job.request_id();
    let mut sequence = 1_u64;
    let mut proposals = Vec::new();
    while let Some(event) = job.recv().await {
        match event {
            QueueEvent::Token { token_id, .. } => {
                metrics.proposal();
                proposals.push(token_id);
                let envelope = Envelope::new(
                    request_id,
                    sequence,
                    Message::Proposal(Proposal {
                        token_ids: vec![token_id],
                    }),
                );
                if send_server(&outbound, peer.clone(), envelope)
                    .await
                    .is_err()
                {
                    job.cancel();
                    return;
                }
                sequence += 1;
            }
            QueueEvent::Complete { .. } => {
                metrics.completion();
                let digest = proposal_digest(&proposals);
                tracing::info!(
                    worker_id = identity.worker.worker_id,
                    dp_rank = identity.worker.dp_rank,
                    draft_incarnation = identity.draft_incarnation_id,
                    %request_id,
                    proposal_digest = %digest,
                    proposal_tokens = proposals.len(),
                    "mock speculative draft completed proposal"
                );
                let envelope = Envelope::new(
                    request_id,
                    sequence,
                    Message::Complete(Complete {
                        final_sequence: sequence,
                        proposal_digest: digest,
                    }),
                );
                let _ = send_server(&outbound, peer, envelope).await;
                return;
            }
            QueueEvent::Cancelled { .. } => {
                let _ = send_server_error(
                    &outbound,
                    peer,
                    request_id,
                    sequence,
                    ErrorCode::Cancelled,
                    FailureState::Cancelled,
                )
                .await;
                return;
            }
            QueueEvent::PrefillComplete => {}
            QueueEvent::PrefillStarted => queue_depth.started(),
            QueueEvent::Queued => {}
        }
    }
}

async fn send_server(
    outbound: &mpsc::Sender<ServerOutbound>,
    peer: Vec<u8>,
    envelope: Envelope,
) -> Result<(), ()> {
    outbound
        .send(ServerOutbound { peer, envelope })
        .await
        .map_err(|_| ())
}

async fn send_server_error(
    outbound: &mpsc::Sender<ServerOutbound>,
    peer: Vec<u8>,
    request_id: Uuid,
    sequence: u64,
    code: ErrorCode,
    state: FailureState,
) -> Result<(), ()> {
    send_server(
        outbound,
        peer,
        Envelope::new(
            request_id,
            sequence,
            Message::Error(ErrorPayload::new(code, state)),
        ),
    )
    .await
}

async fn reap_orphans(
    sessions: &mut HashMap<Uuid, ServerSession>,
    lease: Duration,
    identity: &DraftIdentity,
    metrics: &Arc<SpecdecMetrics>,
) {
    let expired: Vec<_> = sessions
        .iter()
        .filter_map(|(request_id, session)| {
            (session.last_heartbeat.elapsed() >= lease).then_some(*request_id)
        })
        .collect();
    for request_id in expired {
        if let Some(session) = sessions.remove(&request_id) {
            metrics.set_active_sessions(sessions.len());
            metrics.orphan_reap_started();
            reap_session(session).await;
            metrics.orphan_reap_finished();
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                active_sessions = sessions.len(),
                "reaped orphaned mock draft session"
            );
        }
    }
}

async fn reap_session(session: ServerSession) {
    session.forwarder.abort();
    let _ = session.forwarder.await;
    session.cancel.cancel();
}

struct ClientInner {
    expected: DraftIdentity,
    config: DraftClientConfig,
    metrics: Arc<SpecdecMetrics>,
    outbound: mpsc::Sender<Vec<u8>>,
    sessions: Arc<DashMap<Uuid, mpsc::Sender<Envelope>>>,
    session_permits: Arc<Semaphore>,
    cancel: CancellationToken,
    reader: Mutex<Option<JoinHandle<()>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    heartbeat: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Clone)]
pub struct DraftClient {
    inner: Arc<ClientInner>,
}

impl DraftClient {
    pub fn same_connection(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn is_closed(&self) -> bool {
        self.inner.cancel.is_cancelled()
    }

    pub fn close(&self) {
        self.inner.cancel.cancel();
        self.inner.sessions.clear();
    }

    pub async fn connect(
        expected: DraftIdentity,
        config: DraftClientConfig,
    ) -> Result<Self, TransportError> {
        Self::connect_with_metrics(expected, config, Arc::new(SpecdecMetrics::default())).await
    }

    pub(crate) async fn connect_with_metrics(
        expected: DraftIdentity,
        config: DraftClientConfig,
        metrics: Arc<SpecdecMetrics>,
    ) -> Result<Self, TransportError> {
        config.validate()?;
        expected.validate().map_err(|_| {
            TransportError::new(
                TransportErrorKind::Identity,
                FailureState::NotStarted,
                "expected draft identity is invalid",
            )
        })?;
        let socket_identity = Uuid::new_v4().to_string();
        let socket = dealer(&Context::new())
            .set_linger(0)
            .set_sndhwm(config.transport_hwm)
            .set_rcvhwm(config.transport_hwm)
            .set_maxmsgsize(MAX_FRAME_BYTES as i64)
            .set_identity(socket_identity.as_bytes())
            .connect(&expected.address)
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::Connect,
                    FailureState::NotStarted,
                    "failed to connect draft transport",
                )
            })?;
        let (sink, stream) = socket.split();
        let cancel = CancellationToken::new();
        let sessions = Arc::new(DashMap::new());
        let (outbound, outbound_rx) = mpsc::channel(config.outbound_capacity);
        let writer_cancel = cancel.clone();
        let writer = tokio::spawn(run_client_writer(sink, outbound_rx, writer_cancel));
        let reader_cancel = cancel.clone();
        let reader_sessions = sessions.clone();
        let reader_expected = expected.clone();
        let reader = tokio::spawn(run_client_reader(
            stream,
            reader_sessions,
            reader_expected,
            reader_cancel,
        ));
        let inner = Arc::new(ClientInner {
            expected: expected.clone(),
            config: config.clone(),
            metrics,
            outbound,
            sessions,
            session_permits: Arc::new(Semaphore::new(config.max_sessions)),
            cancel,
            reader: Mutex::new(Some(reader)),
            writer: Mutex::new(Some(writer)),
            heartbeat: Mutex::new(None),
        });
        let client = Self { inner };
        if let Err(error) = client.handshake().await {
            let _ = client.shutdown().await;
            return Err(error);
        }
        client.start_heartbeat().await;
        Ok(client)
    }

    async fn handshake(&self) -> Result<(), TransportError> {
        let request_id = Uuid::new_v4();
        let (mut receiver, _permit) = self.register_session(request_id)?;
        let result = async {
            self.send(
                Envelope::new(
                    request_id,
                    0,
                    Message::Hello(Hello {
                        expected: self.inner.expected.clone(),
                    }),
                ),
                FailureState::NotStarted,
            )
            .await?;
            let response =
                tokio::time::timeout(self.inner.config.handshake_timeout, receiver.recv())
                    .await
                    .map_err(|_| {
                        TransportError::new(
                            TransportErrorKind::Timeout,
                            FailureState::NotStarted,
                            "draft handshake timed out",
                        )
                    })?
                    .ok_or_else(|| {
                        TransportError::new(
                            TransportErrorKind::Closed,
                            FailureState::NotStarted,
                            "draft handshake channel closed",
                        )
                    })?;
            match response.message {
                Message::HelloAck(ack)
                    if response.sequence == 0 && ack.identity == self.inner.expected =>
                {
                    Ok(())
                }
                Message::Error(error) => Err(remote_error(error)),
                _ => Err(TransportError::new(
                    TransportErrorKind::Identity,
                    FailureState::ProtocolInvalid,
                    "draft handshake identity mismatch",
                )),
            }
        }
        .await;
        self.inner.sessions.remove(&request_id);
        result
    }

    async fn start_heartbeat(&self) {
        let interval = Duration::from_millis(
            (u64::from(self.inner.expected.orphan_cleanup_timeout_ms) / 3).max(10),
        );
        let connection_id = Uuid::new_v4();
        let outbound = self.inner.outbound.clone();
        let cancel = self.inner.cancel.clone();
        let task = tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut sequence = 1_u64;
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    _ = ticker.tick() => {
                        let envelope = Envelope::new(
                            connection_id,
                            sequence,
                            Message::Heartbeat(Heartbeat::default()),
                        );
                        let frame = match envelope.encode() {
                            Ok(frame) => frame,
                            Err(_) => {
                                cancel.cancel();
                                break;
                            }
                        };
                        match outbound.try_send(frame) {
                            Ok(()) => sequence = sequence.saturating_add(1),
                            Err(mpsc::error::TrySendError::Full(_)) => continue,
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                cancel.cancel();
                                break;
                            }
                        }
                    }
                }
            }
        });
        *self.inner.heartbeat.lock().await = Some(task);
    }

    pub async fn start(
        &self,
        request_id: Uuid,
        start: Start,
    ) -> Result<DraftSession, TransportError> {
        let max_output_tokens = start.max_output_tokens as usize;
        let (mut receiver, permit) = self.register_session(request_id)?;
        let mut pending = PendingStart::new(self.clone(), request_id);
        self.send(
            Envelope::new(request_id, 0, Message::Start(start)),
            FailureState::NotStarted,
        )
        .await?;
        tracing::info!(
            worker_id = self.inner.expected.worker.worker_id,
            dp_rank = self.inner.expected.worker.dp_rank,
            draft_incarnation = self.inner.expected.draft_incarnation_id,
            %request_id,
            "draft client queued START"
        );
        pending.mark_sent();
        let mut sequence = SequenceValidator::starting_at(0);
        let response =
            match tokio::time::timeout(self.inner.config.start_timeout, receiver.recv()).await {
                Ok(Some(response)) => response,
                Ok(None) => {
                    return Err(TransportError::new(
                        TransportErrorKind::Closed,
                        FailureState::Ambiguous,
                        "draft START acknowledgement channel closed",
                    ));
                }
                Err(_) => {
                    return Err(TransportError::new(
                        TransportErrorKind::Timeout,
                        FailureState::Ambiguous,
                        "draft START acknowledgement timed out",
                    ));
                }
            };
        tracing::info!(
            worker_id = self.inner.expected.worker.worker_id,
            dp_rank = self.inner.expected.worker.dp_rank,
            draft_incarnation = self.inner.expected.draft_incarnation_id,
            %request_id,
            "draft client start waiter received response"
        );
        if sequence.observe(response.sequence).is_err() {
            return Err(TransportError::new(
                TransportErrorKind::Protocol,
                FailureState::ProtocolInvalid,
                "draft START acknowledgement sequence is invalid",
            ));
        }
        match response.message {
            Message::StartAck(_) => {
                pending.complete();
                Ok(DraftSession {
                    client: self.clone(),
                    request_id,
                    receiver,
                    sequence,
                    proposal_tokens: Vec::new(),
                    max_output_tokens,
                    permit: Some(permit),
                    cleaned: false,
                })
            }
            Message::Error(error) => {
                if error.state == FailureState::NotStarted {
                    self.inner.sessions.remove(&request_id);
                    pending.complete();
                }
                Err(remote_error(error))
            }
            _ => Err(TransportError::new(
                TransportErrorKind::Protocol,
                FailureState::ProtocolInvalid,
                "draft START acknowledgement is invalid",
            )),
        }
    }

    fn register_session(
        &self,
        request_id: Uuid,
    ) -> Result<(mpsc::Receiver<Envelope>, OwnedSemaphorePermit), TransportError> {
        let permit = self
            .inner
            .session_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::Backpressure,
                    FailureState::NotStarted,
                    "draft client session limit is reached",
                )
            })?;
        let (sender, receiver) = mpsc::channel(self.inner.config.session_capacity);
        match self.inner.sessions.entry(request_id) {
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(sender);
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                return Err(TransportError::new(
                    TransportErrorKind::Protocol,
                    FailureState::NotStarted,
                    "request ID is already active",
                ));
            }
        }
        Ok((receiver, permit))
    }

    async fn send(&self, envelope: Envelope, state: FailureState) -> Result<(), TransportError> {
        let frame = envelope.encode().map_err(|_| {
            TransportError::new(
                TransportErrorKind::Protocol,
                state,
                "draft transport message is invalid",
            )
        })?;
        self.inner.outbound.try_send(frame).map_err(|error| {
            let (kind, message) = match error {
                mpsc::error::TrySendError::Full(_) => (
                    TransportErrorKind::Backpressure,
                    "draft transport send queue is full",
                ),
                mpsc::error::TrySendError::Closed(_) => (
                    TransportErrorKind::Closed,
                    "draft transport writer is closed",
                ),
            };
            TransportError::new(kind, state, message)
        })
    }

    pub async fn shutdown(&self) -> Result<(), TransportError> {
        self.inner.cancel.cancel();
        self.inner.sessions.clear();
        let heartbeat = self.inner.heartbeat.lock().await.take();
        let reader = self.inner.reader.lock().await.take();
        let writer = self.inner.writer.lock().await.take();
        join_task(heartbeat)
            .await
            .and(join_task(reader).await)
            .and(join_task(writer).await)
    }
}

struct PendingStart {
    client: DraftClient,
    request_id: Uuid,
    sent: bool,
    complete: bool,
}

impl PendingStart {
    fn new(client: DraftClient, request_id: Uuid) -> Self {
        Self {
            client,
            request_id,
            sent: false,
            complete: false,
        }
    }

    fn mark_sent(&mut self) {
        self.sent = true;
    }

    fn complete(&mut self) {
        self.complete = true;
    }
}

impl Drop for PendingStart {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        if self.sent {
            self.client.close();
        }
        self.client.inner.sessions.remove(&self.request_id);
    }
}

impl Drop for ClientInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn run_client_writer<S>(
    mut sink: S,
    mut outbound: mpsc::Receiver<Vec<u8>>,
    cancel: CancellationToken,
) where
    S: Sink<Multipart> + Unpin,
    S::Error: fmt::Display,
{
    while let Some(frame) = tokio::select! {
        biased;
        _ = cancel.cancelled() => None,
        frame = outbound.recv() => frame,
    } {
        let send = sink.send(Multipart::from(vec![frame]));
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = send => result,
        };
        if let Err(error) = result {
            tracing::warn!(%error, "draft client writer stopped");
            cancel.cancel();
            break;
        }
    }
}

async fn run_client_reader<S>(
    mut stream: S,
    sessions: Arc<DashMap<Uuid, mpsc::Sender<Envelope>>>,
    identity: DraftIdentity,
    cancel: CancellationToken,
) where
    S: Stream<Item = tmq::Result<Multipart>> + Unpin,
{
    loop {
        let incoming = tokio::select! {
            biased;
            _ = cancel.cancelled() => None,
            incoming = stream.next() => incoming,
        };
        let Some(incoming) = incoming else {
            break;
        };
        let mut frames = match incoming {
            Ok(frames) => frames,
            Err(error) => {
                tracing::warn!(%error, "draft client reader stopped");
                cancel.cancel();
                break;
            }
        };
        if frames.len() != 1 {
            tracing::warn!(
                frame_count = frames.len(),
                "rejected malformed DEALER message"
            );
            continue;
        }
        let frame = frames.pop_front().expect("checked frame count");
        let envelope = match Envelope::decode(frame.as_ref()) {
            Ok(envelope) => envelope,
            Err(error) => {
                tracing::warn!(%error, "rejected invalid client protocol frame");
                continue;
            }
        };
        let request_id = envelope.request_id;
        if matches!(&envelope.message, Message::StartAck(_)) {
            tracing::info!(
                worker_id = identity.worker.worker_id,
                dp_rank = identity.worker.dp_rank,
                draft_incarnation = identity.draft_incarnation_id,
                %request_id,
                "draft client reader received START_ACK"
            );
        }
        let sender = sessions.get(&request_id).map(|sender| sender.clone());
        if let Some(sender) = sender {
            if let Err(error) = sender.try_send(envelope) {
                tracing::warn!(
                    %request_id,
                    %error,
                    "draft client session response queue rejected a protocol frame"
                );
                sessions.remove(&request_id);
            }
        } else if !matches!(envelope.message, Message::HeartbeatAck(_)) {
            tracing::warn!(
                %request_id,
                "draft client received a protocol frame for an unknown session"
            );
        }
    }
    cancel.cancel();
    sessions.clear();
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftProposal {
    pub token_ids: Vec<u32>,
    pub proposal_digest: String,
}

pub struct DraftSession {
    client: DraftClient,
    request_id: Uuid,
    receiver: mpsc::Receiver<Envelope>,
    sequence: SequenceValidator,
    proposal_tokens: Vec<u32>,
    max_output_tokens: usize,
    permit: Option<OwnedSemaphorePermit>,
    cleaned: bool,
}

impl DraftSession {
    pub fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub async fn collect(&mut self) -> Result<DraftProposal, TransportError> {
        loop {
            let envelope = tokio::time::timeout(
                self.client.inner.config.inactivity_timeout,
                self.receiver.recv(),
            )
            .await
            .map_err(|_| {
                TransportError::new(
                    TransportErrorKind::Timeout,
                    FailureState::Ambiguous,
                    "draft proposal stream timed out",
                )
            })?
            .ok_or_else(|| {
                TransportError::new(
                    TransportErrorKind::Closed,
                    FailureState::Ambiguous,
                    "draft proposal stream closed",
                )
            })?;
            self.sequence.observe(envelope.sequence).map_err(|_| {
                TransportError::new(
                    TransportErrorKind::Protocol,
                    FailureState::ProtocolInvalid,
                    "draft proposal sequence is invalid",
                )
            })?;
            match envelope.message {
                Message::Proposal(proposal) => {
                    let total = self
                        .proposal_tokens
                        .len()
                        .checked_add(proposal.token_ids.len())
                        .ok_or_else(|| {
                            TransportError::new(
                                TransportErrorKind::Protocol,
                                FailureState::ProtocolInvalid,
                                "draft proposal token count overflowed",
                            )
                        })?;
                    if total > self.max_output_tokens {
                        return Err(TransportError::new(
                            TransportErrorKind::Protocol,
                            FailureState::ProtocolInvalid,
                            "draft proposal exceeds the requested token limit",
                        ));
                    }
                    self.proposal_tokens.extend(proposal.token_ids);
                }
                Message::Complete(complete) => {
                    let digest = proposal_digest(&self.proposal_tokens);
                    if complete.proposal_digest != digest {
                        return Err(TransportError::new(
                            TransportErrorKind::Protocol,
                            FailureState::ProtocolInvalid,
                            "draft proposal digest is invalid",
                        ));
                    }
                    return Ok(DraftProposal {
                        token_ids: self.proposal_tokens.clone(),
                        proposal_digest: digest,
                    });
                }
                Message::Error(error) => return Err(remote_error(error)),
                _ => {
                    return Err(TransportError::new(
                        TransportErrorKind::Protocol,
                        FailureState::ProtocolInvalid,
                        "unexpected draft proposal message",
                    ));
                }
            }
        }
    }

    pub async fn cleanup(&mut self) -> Result<(), TransportError> {
        let sequence = self.sequence.next();
        let send_result = self
            .client
            .send(
                Envelope::new(
                    self.request_id,
                    sequence,
                    Message::Cleanup(Cleanup::default()),
                ),
                FailureState::Accepted,
            )
            .await;
        let result = match send_result {
            Ok(()) => match tokio::time::timeout(self.client.inner.config.cleanup_timeout, async {
                while let Some(envelope) = self.receiver.recv().await {
                    if envelope.sequence != sequence {
                        continue;
                    }
                    match envelope.message {
                        Message::CleanupAck(_) => {
                            self.client.inner.metrics.cleanup_acknowledged();
                            tracing::info!(
                                worker_id = self.client.inner.expected.worker.worker_id,
                                dp_rank = self.client.inner.expected.worker.dp_rank,
                                draft_incarnation = self.client.inner.expected.draft_incarnation_id,
                                request_id = %self.request_id,
                                "mock speculative target received draft cleanup acknowledgement"
                            );
                            return Ok(());
                        }
                        Message::Error(error) => return Err(remote_error(error)),
                        Message::Proposal(_) | Message::Complete(_) => continue,
                        _ => {
                            return Err(TransportError::new(
                                TransportErrorKind::Protocol,
                                FailureState::ProtocolInvalid,
                                "draft cleanup acknowledgement is invalid",
                            ));
                        }
                    }
                }
                Err(TransportError::new(
                    TransportErrorKind::Closed,
                    FailureState::Ambiguous,
                    "draft cleanup channel closed",
                ))
            })
            .await
            {
                Ok(result) => result,
                Err(_) => Err(TransportError::new(
                    TransportErrorKind::Timeout,
                    FailureState::Ambiguous,
                    "draft cleanup timed out",
                )),
            },
            Err(error) => Err(error),
        };
        self.client.inner.sessions.remove(&self.request_id);
        self.permit.take();
        self.cleaned = true;
        if result
            .as_ref()
            .is_err_and(|error| error.kind == TransportErrorKind::Timeout)
        {
            self.client.inner.metrics.cleanup_timed_out();
        }
        result
    }

    fn try_send_cleanup(&self) {
        let envelope = Envelope::new(
            self.request_id,
            self.sequence.next(),
            Message::Cleanup(Cleanup::default()),
        );
        if let Ok(frame) = envelope.encode() {
            let _ = self.client.inner.outbound.try_send(frame);
        }
    }
}

impl Drop for DraftSession {
    fn drop(&mut self) {
        if !self.cleaned {
            self.try_send_cleanup();
            self.client.inner.sessions.remove(&self.request_id);
        }
    }
}

fn remote_error(error: ErrorPayload) -> TransportError {
    let kind = match error.code {
        ErrorCode::IdentityMismatch => TransportErrorKind::Identity,
        ErrorCode::QueueFull => TransportErrorKind::Queue,
        ErrorCode::InvalidMessage | ErrorCode::DuplicateRequest | ErrorCode::UnknownRequest => {
            TransportErrorKind::Protocol
        }
        ErrorCode::Cancelled => TransportErrorKind::Closed,
        ErrorCode::Internal => TransportErrorKind::Task,
    };
    TransportError::new(kind, error.state, "draft rejected the request")
}

fn queue_transport_error(error: QueueError) -> TransportError {
    let kind = if error == QueueError::Full {
        TransportErrorKind::Backpressure
    } else {
        TransportErrorKind::Queue
    };
    TransportError::new(kind, FailureState::NotStarted, "draft queue failed")
}

async fn join_task(task: Option<JoinHandle<()>>) -> Result<(), TransportError> {
    let Some(mut task) = task else {
        return Ok(());
    };
    match tokio::time::timeout(TASK_SHUTDOWN_TIMEOUT, &mut task).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(TransportError::new(
            TransportErrorKind::Task,
            FailureState::Ambiguous,
            "draft transport task failed",
        )),
        Err(_) => {
            task.abort();
            let _ = task.await;
            Err(TransportError::new(
                TransportErrorKind::Timeout,
                FailureState::Ambiguous,
                "draft transport task shutdown timed out",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{EndpointId, WorkerWithDpRank};

    use super::*;
    use crate::specdec::PROTOCOL;

    #[test]
    fn sequence_validator_rejects_duplicate_and_out_of_order_proposals() {
        let mut validator = SequenceValidator::starting_at(1);
        validator.observe(1).unwrap();
        assert!(validator.observe(1).is_err());
        assert!(validator.observe(3).is_err());
    }

    #[tokio::test]
    async fn reaper_deadline_is_the_exact_oldest_live_lease() {
        let lease = Duration::from_millis(500);
        let now = Instant::now();
        let peer = vec![1];
        let forwarder = tokio::spawn(std::future::pending::<()>());
        let mut sessions = HashMap::new();
        sessions.insert(
            Uuid::from_u128(1),
            ServerSession {
                peer: peer.clone(),
                last_heartbeat: now,
                cancel: CancellationToken::new(),
                forwarder,
            },
        );
        let mut authenticated_peers = HashMap::new();
        authenticated_peers.insert(peer, now - Duration::from_millis(100));

        assert_eq!(
            next_reap_deadline(&sessions, &authenticated_peers, lease),
            now + lease
        );

        let forwarder = sessions.remove(&Uuid::from_u128(1)).unwrap().forwarder;
        forwarder.abort();
        let _ = forwarder.await;
        assert_eq!(
            next_reap_deadline(&sessions, &authenticated_peers, lease),
            now + Duration::from_millis(400)
        );
    }

    #[tokio::test]
    async fn clean_client_reader_eof_cancels_connection_and_clears_sessions() {
        let sessions = Arc::new(DashMap::new());
        let (sender, _receiver) = mpsc::channel(1);
        sessions.insert(Uuid::from_u128(1), sender);
        let cancel = CancellationToken::new();

        run_client_reader(
            futures::stream::empty::<tmq::Result<Multipart>>(),
            sessions.clone(),
            DraftIdentity {
                endpoint: EndpointId::from("specdec/draft/generate"),
                worker: WorkerWithDpRank::new(17, 0),
                draft_incarnation_id: 23,
                protocol: PROTOCOL.to_string(),
                address: "tcp://127.0.0.1:1".to_string(),
                orphan_cleanup_timeout_ms: 500,
            },
            cancel.clone(),
        )
        .await;

        assert!(cancel.is_cancelled());
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn disconnected_peer_response_does_not_break_the_next_real_client() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("tcp://{}", listener.local_addr().unwrap());
        drop(listener);
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 200,
        };
        let server = DraftServer::bind(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_millis(100),
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig::default(),
            },
            identity.clone(),
        )
        .await
        .unwrap();
        let first = DraftClient::connect(identity.clone(), DraftClientConfig::default())
            .await
            .unwrap();
        let first_session = first
            .start(
                Uuid::from_u128(31),
                Start {
                    prompt_token_ids: vec![1, 2],
                    max_output_tokens: 2,
                },
            )
            .await
            .unwrap();
        first.close();
        drop(first_session);
        first.shutdown().await.unwrap();
        tokio::time::sleep(Duration::from_millis(150)).await;

        let second = DraftClient::connect(identity, DraftClientConfig::default())
            .await
            .unwrap();
        let mut second_session = second
            .start(
                Uuid::from_u128(32),
                Start {
                    prompt_token_ids: vec![3, 4],
                    max_output_tokens: 2,
                },
            )
            .await
            .unwrap();
        let proposal = second_session.collect().await.unwrap();
        assert_eq!(proposal.token_ids, vec![3, 4]);
        second_session.cleanup().await.unwrap();

        second.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn cleanup_timeout_updates_metric_and_releases_session_admission() {
        let request_id = Uuid::from_u128(2);
        let expected = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: "tcp://127.0.0.1:1".to_string(),
            orphan_cleanup_timeout_ms: 500,
        };
        let metrics = Arc::new(SpecdecMetrics::default());
        let (outbound, _outbound_rx) = mpsc::channel(1);
        let sessions = Arc::new(DashMap::new());
        let (response_tx, response_rx) = mpsc::channel(1);
        sessions.insert(request_id, response_tx);
        let session_permits = Arc::new(Semaphore::new(1));
        let permit = session_permits.clone().acquire_owned().await.unwrap();
        let client = DraftClient {
            inner: Arc::new(ClientInner {
                expected,
                config: DraftClientConfig {
                    cleanup_timeout: Duration::from_millis(1),
                    ..DraftClientConfig::default()
                },
                metrics: metrics.clone(),
                outbound,
                sessions,
                session_permits: session_permits.clone(),
                cancel: CancellationToken::new(),
                reader: Mutex::new(None),
                writer: Mutex::new(None),
                heartbeat: Mutex::new(None),
            }),
        };
        let mut session = DraftSession {
            client,
            request_id,
            receiver: response_rx,
            sequence: SequenceValidator::starting_at(1),
            proposal_tokens: Vec::new(),
            max_output_tokens: 1,
            permit: Some(permit),
            cleaned: false,
        };

        let error = session.cleanup().await.unwrap_err();

        assert_eq!(error.kind, TransportErrorKind::Timeout);
        assert_eq!(metrics.snapshot().cleanup_timeouts, 1);
        assert_eq!(session_permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn server_metrics_follow_success_rejection_and_bounded_admission() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("tcp://{}", listener.local_addr().unwrap());
        drop(listener);
        let identity = DraftIdentity {
            endpoint: EndpointId::from("specdec/draft/generate"),
            worker: WorkerWithDpRank::new(17, 0),
            draft_incarnation_id: 23,
            protocol: PROTOCOL.to_string(),
            address: address.clone(),
            orphan_cleanup_timeout_ms: 500,
        };
        let prompt = vec![1, 2, 3];
        let metrics = Arc::new(SpecdecMetrics::default());
        let server = DraftServer::bind_with_metrics(
            DraftServerConfig {
                bind_address: address,
                transport_hwm: 64,
                outbound_capacity: 64,
                prefill_duration: Duration::from_secs(10),
                token_interval: Duration::ZERO,
                token_mode: TokenMode::Echo,
                scheduler: SchedulerConfig {
                    queue_capacity: 1,
                    concurrency: 1,
                    output_capacity: 8,
                },
            },
            identity.clone(),
            metrics.clone(),
        )
        .await
        .unwrap();
        let client = DraftClient::connect_with_metrics(
            identity,
            DraftClientConfig {
                max_sessions: 8,
                ..DraftClientConfig::default()
            },
            metrics.clone(),
        )
        .await
        .unwrap();

        let mut first = client
            .start(
                Uuid::from_u128(13),
                Start {
                    prompt_token_ids: prompt.clone(),
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let mut second = client
            .start(
                Uuid::from_u128(14),
                Start {
                    prompt_token_ids: prompt.clone(),
                    max_output_tokens: 1,
                },
            )
            .await
            .unwrap();
        let rejected = client
            .start(
                Uuid::from_u128(15),
                Start {
                    prompt_token_ids: prompt,
                    max_output_tokens: 1,
                },
            )
            .await;
        let Err(rejected) = rejected else {
            panic!("bounded draft admission unexpectedly accepted a third session");
        };
        assert_eq!(rejected.kind, TransportErrorKind::Queue);
        first.cleanup().await.unwrap();
        second.cleanup().await.unwrap();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.starts_accepted, 2);
        assert_eq!(snapshot.starts_rejected, 1);
        assert_eq!(snapshot.cleanup_acknowledgements, 4);
        assert_eq!(snapshot.active_sessions, 0);
        assert_eq!(snapshot.queue_rejections, 1);
        client.shutdown().await.unwrap();
        server.shutdown().await.unwrap();
    }
}
