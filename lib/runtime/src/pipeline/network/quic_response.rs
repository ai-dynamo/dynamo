// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-lane QUIC transport for streaming worker responses.
//!
//! A worker keeps a fixed connection bundle to a frontend and distributes a
//! fixed set of long-lived bidirectional streams (lanes) across it. Logical
//! response frames are request-hashed to a lane once, queued in a bounded
//! Tokio channel, and written in batches by the lane's sole writer task.

use std::{
    collections::{HashMap, VecDeque},
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use arc_swap::ArcSwapOption;
use bytes::{BufMut, Bytes, BytesMut};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use parking_lot::Mutex;
use prometheus::IntCounter;
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use rustls::{
    DigitallySignedStruct, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::{
    io::{AsyncRead, AsyncReadExt, BufReader},
    sync::{Notify, Semaphore, mpsc, oneshot},
    time::{Instant, sleep_until},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

use super::{ConnectionInfo, RegisteredStream, StreamReceiver};
use crate::{
    config::environment_names::quic_response, discovery::EndpointInstanceId,
    engine::AsyncEngineContext, pipeline::PipelineError,
};

pub const TRANSPORT_NAME: &str = "quic-response";
const PROTOCOL_VERSION: u8 = 3;
const ALPN: &[u8] = b"dynamo-response-v2";
const BULK_CONNECTIONS: usize = 8;
const BULK_LANES: usize = 8;
const PRIORITY_CONNECTIONS: usize = 1;
const LANE_QUEUE_CAPACITY: usize = 512;
// Bulk writers can spend tens of milliseconds waiting for Quinn to accept a
// large batch. Keep enough bounded slack to absorb those stalls without
// parking every response task on the lane semaphore. Priority traffic stays
// on the smaller queue because it never carries the long token stream.
const BULK_LANE_QUEUE_CAPACITY: usize = 4_096;
const MAX_BATCH_FRAMES: usize = 512;
const DEFAULT_BATCH_INTERVAL_US: u64 = 5_000;
const MAX_BATCH_INTERVAL_US: u64 = 100_000;
const FRAME_HEADER_LEN: usize = 1 + 16 + 4;
// A writer batch can contain hundreds of small response frames. Buffer reads
// at the receiver so parsing those frames does not poll Quinn once per header
// and payload and exhaust Tokio's cooperative task budget.
const RECEIVE_BUFFER_CAPACITY: usize = 256 * 1024;
// A single frontend UDP socket is limited by the host receive-buffer ceiling.
// Reuse-port endpoints preserve one advertised address while spreading QUIC
// connections and receive queues across several sockets.
#[cfg(target_os = "linux")]
const SERVER_ENDPOINTS: usize = 8;
#[cfg(not(target_os = "linux"))]
const SERVER_ENDPOINTS: usize = 1;
const MAX_FRAME_PAYLOAD: usize = 32 * 1024 * 1024;
const RESPONSE_BUFFER_CAPACITY: usize = 64;
const MAX_RESPONSE_BUFFER_CAPACITY: usize = 65_536;
const MAX_DEFERRED_RESPONSE_FRAMES: usize = 1_024;
const MAX_DEFERRED_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const TOMBSTONE_TTL: Duration = Duration::from_secs(5);
const CLOSE_CODE_INVARIANT: quinn::VarInt = quinn::VarInt::from_u32(1);

#[derive(Debug, Clone, Copy)]
pub struct QuicResponseConfig {
    pub batch_interval: Duration,
    pub response_buffer_capacity: usize,
}

impl QuicResponseConfig {
    pub fn from_env() -> Result<Self, PipelineError> {
        let interval_us = parse_env_range(
            quic_response::DYN_QUIC_RESPONSE_BATCH_INTERVAL_US,
            DEFAULT_BATCH_INTERVAL_US,
            0,
            MAX_BATCH_INTERVAL_US,
        )?;
        let response_buffer_capacity = parse_env_range(
            quic_response::DYN_QUIC_RESPONSE_BUFFER_CAPACITY,
            RESPONSE_BUFFER_CAPACITY,
            1,
            MAX_RESPONSE_BUFFER_CAPACITY,
        )?;
        Ok(Self {
            batch_interval: Duration::from_micros(interval_us),
            response_buffer_capacity,
        })
    }
}

fn parse_env_range<T>(name: &str, default: T, min: T, max: T) -> Result<T, PipelineError>
where
    T: Copy + std::str::FromStr + PartialOrd + fmt::Display,
{
    let Some(raw) = std::env::var(name).ok().filter(|value| !value.is_empty()) else {
        return Ok(default);
    };
    let value = raw
        .parse::<T>()
        .map_err(|_| PipelineError::Generic(format!("invalid {name}: '{raw}' is not a number")))?;
    if value < min || value > max {
        return Err(PipelineError::Generic(format!(
            "invalid {name}: {value} is outside {min}..={max}"
        )));
    }
    Ok(value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicResponseConnectionInfo {
    version: u8,
    address: String,
    frontend_id: String,
    registration_id: Uuid,
    request_id: String,
    certificate_sha256: String,
}

impl From<QuicResponseConnectionInfo> for ConnectionInfo {
    fn from(value: QuicResponseConnectionInfo) -> Self {
        Self {
            transport: TRANSPORT_NAME.to_string(),
            info: serde_json::to_string(&value).expect("QUIC connection info must serialize"),
        }
    }
}

impl TryFrom<ConnectionInfo> for QuicResponseConnectionInfo {
    type Error = anyhow::Error;

    fn try_from(value: ConnectionInfo) -> Result<Self> {
        if value.transport != TRANSPORT_NAME {
            bail!(
                "expected {TRANSPORT_NAME} connection info, got {}",
                value.transport
            );
        }
        let info: Self = serde_json::from_str(&value.info)?;
        if info.version != PROTOCOL_VERSION {
            bail!(
                "unsupported QUIC response protocol version {} (expected {})",
                info.version,
                PROTOCOL_VERSION
            );
        }
        Ok(info)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum FrameKind {
    Prologue = 1,
    Data = 2,
    Error = 3,
    End = 4,
    Stop = 5,
    Kill = 6,
    Reset = 7,
    FirstData = 8,
    PriorityEnd = 9,
    Bundle = 10,
    Register = 11,
    Registered = 12,
}

impl TryFrom<u8> for FrameKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Prologue),
            2 => Ok(Self::Data),
            3 => Ok(Self::Error),
            4 => Ok(Self::End),
            5 => Ok(Self::Stop),
            6 => Ok(Self::Kill),
            7 => Ok(Self::Reset),
            8 => Ok(Self::FirstData),
            9 => Ok(Self::PriorityEnd),
            10 => Ok(Self::Bundle),
            11 => Ok(Self::Register),
            12 => Ok(Self::Registered),
            _ => bail!("unknown QUIC response frame kind {value}"),
        }
    }
}

#[derive(Debug)]
struct Frame {
    kind: FrameKind,
    registration_id: Uuid,
    payload: Bytes,
}

impl Frame {
    fn new(kind: FrameKind, registration_id: Uuid, payload: Bytes) -> Self {
        Self {
            kind,
            registration_id,
            payload,
        }
    }

    fn header(&self) -> Bytes {
        let mut header = BytesMut::with_capacity(FRAME_HEADER_LEN);
        header.put_u8(self.kind as u8);
        header.extend_from_slice(self.registration_id.as_bytes());
        header.put_u32(self.payload.len() as u32);
        header.freeze()
    }
}

async fn read_frame<R>(recv: &mut R) -> Result<Frame>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0_u8; FRAME_HEADER_LEN];
    recv.read_exact(&mut header).await?;
    let kind = FrameKind::try_from(header[0])?;
    let registration_id = Uuid::from_slice(&header[1..17])?;
    let payload_len = u32::from_be_bytes(header[17..21].try_into().unwrap()) as usize;
    if payload_len > MAX_FRAME_PAYLOAD {
        bail!("QUIC response frame payload {payload_len} exceeds {MAX_FRAME_PAYLOAD}");
    }
    let mut payload = BytesMut::zeroed(payload_len);
    recv.read_exact(&mut payload).await?;
    Ok(Frame {
        kind,
        registration_id,
        payload: payload.freeze(),
    })
}

struct LaneSender(Arc<LaneQueue>);

struct LaneReceiver(Arc<LaneQueue>);

struct LaneQueue {
    frames: SegQueue<Frame>,
    slots: Semaphore,
    ready: Notify,
    scheduled: AtomicBool,
    closed: AtomicBool,
    senders: AtomicUsize,
    #[cfg(test)]
    notifications: AtomicUsize,
}

#[derive(Debug)]
enum LaneTrySendError {
    Closed(Frame),
    Full(Frame),
}

impl LaneQueue {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            frames: SegQueue::new(),
            slots: Semaphore::new(capacity),
            ready: Notify::new(),
            scheduled: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            senders: AtomicUsize::new(1),
            #[cfg(test)]
            notifications: AtomicUsize::new(0),
        })
    }

    fn push_with_permit(
        &self,
        frame: Frame,
        permit: tokio::sync::SemaphorePermit<'_>,
    ) -> std::result::Result<(), Frame> {
        if self.closed.load(Ordering::Acquire) {
            return Err(frame);
        }
        self.frames.push(frame);
        permit.forget();
        if !self.scheduled.swap(true, Ordering::AcqRel) {
            #[cfg(test)]
            self.notifications.fetch_add(1, Ordering::Relaxed);
            self.ready.notify_one();
        }
        Ok(())
    }

    fn try_send(&self, frame: Frame) -> std::result::Result<(), LaneTrySendError> {
        let permit = match self.slots.try_acquire() {
            Ok(permit) => permit,
            Err(tokio::sync::TryAcquireError::Closed) => {
                return Err(LaneTrySendError::Closed(frame));
            }
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                return Err(LaneTrySendError::Full(frame));
            }
        };
        self.push_with_permit(frame, permit)
            .map_err(LaneTrySendError::Closed)
    }

    async fn send(&self, frame: Frame) -> Result<()> {
        let permit = self
            .slots
            .acquire()
            .await
            .map_err(|_| anyhow!("QUIC response lane closed"))?;
        self.push_with_permit(frame, permit)
            .map_err(|_| anyhow!("QUIC response lane closed"))
    }

    fn drain(&self, batch: &mut Vec<Frame>, limit: usize) -> usize {
        let initial_len = batch.len();
        while batch.len() - initial_len < limit {
            let Some(frame) = self.frames.pop() else {
                break;
            };
            batch.push(frame);
        }
        let count = batch.len() - initial_len;
        if count != 0 {
            self.slots.add_permits(count);
        }
        count
    }

    async fn recv_many(&self, batch: &mut Vec<Frame>, limit: usize) -> usize {
        loop {
            let mut notified = Box::pin(self.ready.notified());
            notified.as_mut().enable();
            let count = self.drain(batch, limit);
            if count != 0 {
                return count;
            }
            if self.closed.load(Ordering::Acquire) {
                return 0;
            }
            // Producers suppress notifications while the writer is active. Mark
            // it idle after observing an empty queue, then recheck to close the
            // producer/consumer race without a lock.
            self.scheduled.store(false, Ordering::Release);
            if !self.frames.is_empty() {
                self.scheduled.store(true, Ordering::Release);
                continue;
            }
            notified.await;
        }
    }

    fn close_sender(&self) {
        self.closed.store(true, Ordering::Release);
        self.slots.close();
        self.ready.notify_waiters();
    }

    fn close_receiver(&self) {
        self.closed.store(true, Ordering::Release);
        let mut discarded = 0;
        while self.frames.pop().is_some() {
            discarded += 1;
        }
        if discarded != 0 {
            self.slots.add_permits(discarded);
        }
        self.slots.close();
        self.ready.notify_waiters();
    }
}

impl Drop for LaneReceiver {
    fn drop(&mut self) {
        self.0.close_receiver();
    }
}

impl Clone for LaneSender {
    fn clone(&self) -> Self {
        self.0.senders.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}

impl Drop for LaneSender {
    fn drop(&mut self) {
        if self.0.senders.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.close_sender();
        }
    }
}

impl LaneReceiver {
    async fn recv_many(&mut self, batch: &mut Vec<Frame>, limit: usize) -> usize {
        self.0.recv_many(batch, limit).await
    }

    #[cfg(test)]
    async fn recv(&mut self) -> Option<Frame> {
        let mut batch = Vec::with_capacity(1);
        (self.recv_many(&mut batch, 1).await == 1).then(|| batch.pop().unwrap())
    }
}

impl LaneSender {
    fn try_send(&self, frame: Frame) -> std::result::Result<(), LaneTrySendError> {
        self.0.try_send(frame)
    }

    async fn send(&self, frame: Frame) -> Result<()> {
        self.0.send(frame).await
    }
}

fn lane_queue(capacity: usize) -> (LaneSender, LaneReceiver) {
    let queue = LaneQueue::new(capacity);
    (LaneSender(queue.clone()), LaneReceiver(queue))
}

/// Fill `batch` with up to `max_batch_frames` frames. The first `recv_many`
/// waits while idle;
/// subsequent cancellation-safe `recv_many` calls race the single batch
/// deadline. A zero interval therefore drains an already-ready burst but does
/// not intentionally wait for more work.
async fn receive_batch(
    receiver: &mut LaneReceiver,
    batch: &mut Vec<Frame>,
    interval: Duration,
    max_batch_frames: usize,
) -> bool {
    batch.clear();
    if receiver.recv_many(batch, max_batch_frames).await == 0 {
        return false;
    }

    if interval.is_zero() {
        return true;
    }
    let deadline = Instant::now() + interval;
    let sleep = sleep_until(deadline);
    tokio::pin!(sleep);
    while batch.len() < max_batch_frames {
        tokio::select! {
            biased;
            _ = &mut sleep => break,
            count = receiver.recv_many(batch, max_batch_frames - batch.len()) => {
                if count == 0 {
                    break;
                }
            }
        }
    }
    true
}

const SERVER_STATE_SHARDS: usize = 64;

#[derive(Default)]
struct RegistrationShard {
    pending: HashMap<Uuid, PendingResponse>,
    active: HashMap<Uuid, ActiveResponse>,
}

#[derive(Default)]
struct ServerIndexes {
    registration_instance: HashMap<Uuid, EndpointInstanceId>,
    instance_registrations: HashMap<EndpointInstanceId, Vec<Uuid>>,
    removed_instances: HashMap<EndpointInstanceId, Instant>,
    connection_bundles: HashMap<usize, Uuid>,
    bundle_connections: HashMap<Uuid, HashMap<usize, quinn::Connection>>,
}

struct ServerState {
    registrations: [Mutex<RegistrationShard>; SERVER_STATE_SHARDS],
    indexes: Mutex<ServerIndexes>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            registrations: std::array::from_fn(|_| Mutex::new(RegistrationShard::default())),
            indexes: Mutex::new(ServerIndexes::default()),
        }
    }
}

impl ServerState {
    fn registration(&self, registration_id: Uuid) -> &Mutex<RegistrationShard> {
        &self.registrations[registration_id.as_u128() as usize % SERVER_STATE_SHARDS]
    }
}

#[derive(Default)]
struct DeferredResponse {
    data: VecDeque<Bytes>,
    bytes: usize,
    end: bool,
}

impl DeferredResponse {
    fn push(&mut self, payload: Bytes) -> Result<(), Bytes> {
        let Some(bytes) = self.bytes.checked_add(payload.len()) else {
            return Err(payload);
        };
        if self.data.len() >= MAX_DEFERRED_RESPONSE_FRAMES || bytes > MAX_DEFERRED_RESPONSE_BYTES {
            return Err(payload);
        }
        self.bytes = bytes;
        self.data.push_back(payload);
        Ok(())
    }

    fn take_data(&mut self) -> VecDeque<Bytes> {
        self.bytes = 0;
        std::mem::take(&mut self.data)
    }
}

struct PendingResponse {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamReceiver, String>>,
    bundle_id: Option<Uuid>,
    deferred: DeferredResponse,
    monitor_cancel: CancellationToken,
    monitor_response: Option<oneshot::Sender<mpsc::Sender<Bytes>>>,
}

struct ActiveResponse {
    sender: mpsc::Sender<Bytes>,
    monitor_cancel: CancellationToken,
    bundle_id: Uuid,
    priority_ready: bool,
    priority_draining: bool,
    deferred: DeferredResponse,
}

fn prune_tombstones(indexes: &mut ServerIndexes, now: Instant) {
    indexes
        .removed_instances
        .retain(|_, inserted| now.saturating_duration_since(*inserted) < TOMBSTONE_TTL);
}

pub struct QuicResponseServer {
    endpoints: Arc<[quinn::Endpoint]>,
    advertised_address: SocketAddr,
    frontend_id: String,
    certificate_sha256: String,
    state: Arc<ServerState>,
    response_buffer_capacity: usize,
    shutdown: CancellationToken,
}

impl fmt::Debug for QuicResponseServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuicResponseServer")
            .field("address", &self.advertised_address)
            .field("frontend_id", &self.frontend_id)
            .finish_non_exhaustive()
    }
}

impl QuicResponseServer {
    pub fn new(
        bind_address: SocketAddr,
        advertised_address: SocketAddr,
        shutdown: CancellationToken,
    ) -> Result<Arc<Self>, PipelineError> {
        Self::new_with_stream_window(bind_address, advertised_address, shutdown, None)
    }

    fn new_with_stream_window(
        bind_address: SocketAddr,
        advertised_address: SocketAddr,
        shutdown: CancellationToken,
        stream_receive_window: Option<u32>,
    ) -> Result<Arc<Self>, PipelineError> {
        let response_config = QuicResponseConfig::from_env()?;
        Self::new_with_config(
            bind_address,
            advertised_address,
            shutdown,
            stream_receive_window,
            response_config,
        )
    }

    fn new_with_config(
        bind_address: SocketAddr,
        advertised_address: SocketAddr,
        shutdown: CancellationToken,
        stream_receive_window: Option<u32>,
        response_config: QuicResponseConfig,
    ) -> Result<Arc<Self>, PipelineError> {
        let certified =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).map_err(|error| {
                PipelineError::Generic(format!("failed generating QUIC certificate: {error}"))
            })?;
        let cert_der = CertificateDer::from(certified.cert);
        let fingerprint = Sha256::digest(cert_der.as_ref());
        let certificate_sha256 = encode_hex(&fingerprint);
        let key = PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der());

        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut crypto = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| {
                PipelineError::Generic(format!("failed configuring QUIC TLS: {error}"))
            })?
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key.into())
            .map_err(|error| {
                PipelineError::Generic(format!("failed configuring QUIC certificate: {error}"))
            })?;
        crypto.alpn_protocols = vec![ALPN.to_vec()];

        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(
            QuicServerConfig::try_from(crypto).map_err(|error| {
                PipelineError::Generic(format!("failed configuring QUIC server: {error}"))
            })?,
        ));
        let transport = Arc::get_mut(&mut server_config.transport)
            .expect("new QUIC server config has one transport owner");
        transport.max_concurrent_bidi_streams((BULK_LANES as u32).into());
        transport.max_concurrent_uni_streams(0_u8.into());
        transport.keep_alive_interval(Some(Duration::from_secs(5)));
        if let Some(window) = stream_receive_window {
            transport.stream_receive_window(quinn::VarInt::from_u32(window));
        }

        let socket = bind_server_udp(bind_address, false).map_err(|error| {
            PipelineError::Generic(format!(
                "failed binding QUIC response server socket on {bind_address}: {error}"
            ))
        })?;
        let bound_address = socket.local_addr().map_err(|error| {
            PipelineError::Generic(format!(
                "failed reading QUIC response server socket address: {error}"
            ))
        })?;
        let mut first_socket = Some(socket);
        let mut endpoints = Vec::with_capacity(SERVER_ENDPOINTS);
        for index in 0..SERVER_ENDPOINTS {
            let socket = if index == 0 {
                first_socket
                    .take()
                    .expect("first QUIC response socket exists")
            } else {
                bind_server_udp(bound_address, true).map_err(|error| {
                    PipelineError::Generic(format!(
                        "failed binding QUIC response reuse-port socket {index} on {bound_address}: {error}"
                    ))
                })?
            };
            let endpoint = quinn::Endpoint::new(
                quinn::EndpointConfig::default(),
                Some(server_config.clone()),
                socket,
                Arc::new(quinn::TokioRuntime),
            )
            .map_err(|error| {
                PipelineError::Generic(format!(
                    "failed creating QUIC response endpoint {index} on {bound_address}: {error}"
                ))
            })?;
            endpoints.push(endpoint);
        }
        let endpoints: Arc<[quinn::Endpoint]> = endpoints.into();
        let advertised_address = if advertised_address.port() == 0 {
            bound_address
        } else {
            advertised_address
        };
        let server = Arc::new(Self {
            endpoints,
            advertised_address,
            frontend_id: Uuid::new_v4().to_string(),
            certificate_sha256,
            state: Arc::new(ServerState::default()),
            response_buffer_capacity: response_config.response_buffer_capacity,
            shutdown: shutdown.child_token(),
        });
        Self::spawn_accept_loop(&server);
        Ok(server)
    }

    fn spawn_accept_loop(server: &Arc<Self>) {
        for endpoint in server.endpoints.iter() {
            let endpoint = endpoint.clone();
            let state = server.state.clone();
            let response_buffer_capacity = server.response_buffer_capacity;
            let shutdown = server.shutdown.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = shutdown.cancelled() => {
                            endpoint.close(quinn::VarInt::from_u32(0), b"runtime shutdown");
                            break;
                        }
                        incoming = endpoint.accept() => {
                            let Some(incoming) = incoming else { break };
                            let state = state.clone();
                            tokio::spawn(async move {
                                match incoming.await {
                                    Ok(connection) => run_server_connection(
                                        connection,
                                        state,
                                        response_buffer_capacity,
                                    ).await,
                                    Err(error) => tracing::warn!(%error, "QUIC response handshake failed"),
                                }
                            });
                        }
                    }
                }
            });
        }
    }

    pub fn register_response(
        &self,
        context: Arc<dyn AsyncEngineContext>,
    ) -> RegisteredStream<StreamReceiver> {
        let registration_id = Uuid::new_v4();
        let request_id = context.id().to_string();
        let (pending_tx, pending_rx) = oneshot::channel();
        self.state
            .registration(registration_id)
            .lock()
            .pending
            .insert(
                registration_id,
                PendingResponse {
                    context,
                    connection: pending_tx,
                    bundle_id: None,
                    deferred: DeferredResponse::default(),
                    monitor_cancel: CancellationToken::new(),
                    monitor_response: None,
                },
            );

        let connection_info = QuicResponseConnectionInfo {
            version: PROTOCOL_VERSION,
            address: self.advertised_address.to_string(),
            frontend_id: self.frontend_id.clone(),
            registration_id,
            request_id,
            certificate_sha256: self.certificate_sha256.clone(),
        }
        .into();

        let state = self.state.clone();
        RegisteredStream::new(connection_info, pending_rx)
            .with_registration_id(registration_id)
            .with_cleanup(move || {
                remove_registration(&state, registration_id);
            })
    }

    pub async fn associate_instance(&self, registration_id: Uuid, id: &EndpointInstanceId) -> bool {
        let mut indexes = self.state.indexes.lock();
        let now = Instant::now();
        prune_tombstones(&mut indexes, now);
        if indexes.removed_instances.contains_key(id) {
            drop(indexes);
            remove_registration(&self.state, registration_id);
            return false;
        }
        indexes
            .registration_instance
            .insert(registration_id, id.clone());
        indexes
            .instance_registrations
            .entry(id.clone())
            .or_default()
            .push(registration_id);
        true
    }

    pub async fn cancel_response(&self, registration_id: Uuid) {
        remove_registration(&self.state, registration_id);
    }

    pub async fn cancel_instance_streams(&self, id: &EndpointInstanceId) -> usize {
        let registrations = {
            let mut indexes = self.state.indexes.lock();
            let now = Instant::now();
            prune_tombstones(&mut indexes, now);
            indexes.removed_instances.insert(id.clone(), now);
            let registrations = indexes
                .instance_registrations
                .remove(id)
                .unwrap_or_default();
            for registration_id in &registrations {
                indexes.registration_instance.remove(registration_id);
            }
            registrations
        };
        let count = registrations.len();
        for registration_id in registrations {
            remove_registration_data(&self.state, registration_id);
        }
        count
    }

    pub async fn clear_instance_tombstone(&self, id: &EndpointInstanceId) {
        self.state.indexes.lock().removed_instances.remove(id);
    }
}

impl Drop for QuicResponseServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        for endpoint in self.endpoints.iter() {
            endpoint.close(quinn::VarInt::from_u32(0), b"response server dropped");
        }
    }
}

fn bind_server_udp(address: SocketAddr, join_reuseport: bool) -> std::io::Result<UdpSocket> {
    let socket = Socket::new(
        Domain::for_address(address),
        Type::DGRAM,
        Some(Protocol::UDP),
    )?;
    #[cfg(target_os = "linux")]
    if join_reuseport {
        socket.set_reuse_address(true)?;
        socket.set_reuse_port(true)?;
    }
    socket.set_nonblocking(true)?;
    socket.bind(&address.into())?;
    // Bind the first endpoint exclusively so an ephemeral port cannot join an
    // unrelated server's reuse-port group. Linux permits enabling reuse-port
    // after that first bind, and the remaining seven endpoints can then join.
    #[cfg(target_os = "linux")]
    if !join_reuseport {
        socket.set_reuse_address(true)?;
        socket.set_reuse_port(true)?;
    }
    Ok(socket.into())
}

fn remove_registration_data(state: &ServerState, registration_id: Uuid) {
    let mut registration = state.registration(registration_id).lock();
    if let Some(pending) = registration.pending.remove(&registration_id) {
        pending.monitor_cancel.cancel();
    }
    if let Some(active) = registration.active.remove(&registration_id) {
        active.monitor_cancel.cancel();
    }
}

fn remove_registration(state: &ServerState, registration_id: Uuid) {
    let mut indexes = state.indexes.lock();
    if let Some(instance) = indexes.registration_instance.remove(&registration_id)
        && let Some(registrations) = indexes.instance_registrations.get_mut(&instance)
    {
        registrations.retain(|candidate| *candidate != registration_id);
        if registrations.is_empty() {
            indexes.instance_registrations.remove(&instance);
        }
    }
    drop(indexes);
    remove_registration_data(state, registration_id);
}

fn fail_registration(state: &ServerState, registration_id: Uuid, reason: &str) {
    let mut indexes = state.indexes.lock();
    if let Some(instance) = indexes.registration_instance.remove(&registration_id)
        && let Some(registrations) = indexes.instance_registrations.get_mut(&instance)
    {
        registrations.retain(|candidate| *candidate != registration_id);
        if registrations.is_empty() {
            indexes.instance_registrations.remove(&instance);
        }
    }
    drop(indexes);

    let mut registration = state.registration(registration_id).lock();
    if let Some(pending) = registration.pending.remove(&registration_id) {
        pending.monitor_cancel.cancel();
        let _ = pending.connection.send(Err(reason.to_string()));
    }
    if let Some(active) = registration.active.remove(&registration_id) {
        active.monitor_cancel.cancel();
    }
}

async fn run_server_connection(
    connection: quinn::Connection,
    state: Arc<ServerState>,
    response_buffer_capacity: usize,
) {
    let connection_id = connection.stable_id();
    crate::metrics::quic_response::track_connection(connection.clone());
    tracing::debug!(connection_id, remote = %connection.remote_address(), "QUIC response connection established");
    loop {
        match connection.accept_bi().await {
            Ok((send, recv)) => {
                let lane_connection = connection.clone();
                let lane_state = state.clone();
                tokio::spawn(async move {
                    let result = run_server_lane(
                        send,
                        recv,
                        lane_connection.clone(),
                        lane_state.clone(),
                        response_buffer_capacity,
                    )
                    .await;
                    if let Err(error) = result {
                        tracing::warn!(connection_id, %error, "QUIC response lane failed; closing connection");
                        fail_server_connection(&lane_state, connection_id);
                        lane_connection
                            .close(CLOSE_CODE_INVARIANT, b"response lane invariant failure");
                    }
                });
            }
            Err(error) => {
                if connection.close_reason().is_none() {
                    tracing::warn!(connection_id, %error, "QUIC response connection failed while accepting lane");
                }
                fail_server_connection(&state, connection_id);
                break;
            }
        }
    }
}

fn register_server_connection_bundle(
    state: &ServerState,
    connection: quinn::Connection,
    bundle_id: Uuid,
) -> Result<()> {
    let connection_id = connection.stable_id();
    let mut indexes = state.indexes.lock();
    if let Some(existing) = indexes.connection_bundles.get(&connection_id)
        && *existing != bundle_id
    {
        bail!(
            "QUIC response connection {connection_id} changed bundle id from {existing} to {bundle_id}"
        );
    }
    indexes.connection_bundles.insert(connection_id, bundle_id);
    indexes
        .bundle_connections
        .entry(bundle_id)
        .or_default()
        .insert(connection_id, connection);
    Ok(())
}

fn fail_server_connection(state: &ServerState, connection_id: usize) {
    let (bundle_id, connections) = {
        let mut indexes = state.indexes.lock();
        let Some(bundle_id) = indexes.connection_bundles.remove(&connection_id) else {
            return;
        };
        indexes
            .connection_bundles
            .retain(|_, candidate| *candidate != bundle_id);
        let connections = indexes
            .bundle_connections
            .remove(&bundle_id)
            .unwrap_or_default()
            .into_values()
            .collect::<Vec<_>>();
        (bundle_id, connections)
    };
    crate::metrics::quic_response::record_bundle_failure("frontend");

    for connection in connections {
        connection.close(
            CLOSE_CODE_INVARIANT,
            b"response connection bundle invariant failure",
        );
    }

    let doomed = state
        .registrations
        .iter()
        .flat_map(|shard| {
            let shard = shard.lock();
            shard
                .pending
                .iter()
                .filter_map(|(registration_id, pending)| {
                    (pending.bundle_id == Some(bundle_id)).then_some(*registration_id)
                })
                .chain(shard.active.iter().filter_map(|(registration_id, active)| {
                    (active.bundle_id == bundle_id).then_some(*registration_id)
                }))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for registration_id in doomed {
        fail_registration(
            state,
            registration_id,
            "QUIC response connection bundle failed before stream establishment",
        );
    }
}

async fn run_server_lane(
    mut send: quinn::SendStream,
    recv: quinn::RecvStream,
    connection: quinn::Connection,
    state: Arc<ServerState>,
    response_buffer_capacity: usize,
) -> Result<()> {
    let mut recv = BufReader::with_capacity(RECEIVE_BUFFER_CAPACITY, recv);
    let bundle = read_frame(&mut recv).await?;
    if bundle.kind != FrameKind::Bundle || !bundle.payload.is_empty() {
        bail!("QUIC response lane did not start with a bundle frame");
    }
    let bundle_id = bundle.registration_id;
    register_server_connection_bundle(&state, connection, bundle_id)?;

    let (control_tx, mut control_rx) = mpsc::channel::<Frame>(RESPONSE_BUFFER_CAPACITY);
    let mut writer = tokio::spawn(async move {
        while let Some(frame) = control_rx.recv().await {
            let mut chunks = [frame.header(), frame.payload];
            send.write_all_chunks(&mut chunks).await?;
        }
        Ok::<(), quinn::WriteError>(())
    });

    let reader = async {
        loop {
            let frame = read_frame(&mut recv).await?;
            process_server_frame(
                frame,
                bundle_id,
                &state,
                &control_tx,
                response_buffer_capacity,
            )
            .await?;
        }
        #[allow(unreachable_code)]
        Ok::<(), anyhow::Error>(())
    };
    tokio::pin!(reader);
    tokio::select! {
        result = &mut reader => {
            writer.abort();
            let _ = writer.await;
            result
        }
        result = &mut writer => match result {
            Ok(Ok(())) => bail!("QUIC reverse-control writer exited unexpectedly"),
            Ok(Err(error)) => Err(error.into()),
            Err(error) => Err(error.into()),
        },
    }
}

async fn process_server_frame(
    frame: Frame,
    bundle_id: Uuid,
    state: &Arc<ServerState>,
    control_tx: &mpsc::Sender<Frame>,
    response_buffer_capacity: usize,
) -> Result<()> {
    match frame.kind {
        FrameKind::Register => {
            if !frame.payload.is_empty() {
                bail!("QUIC response Register frame carried a payload");
            }
            let monitor = {
                let mut registration = state.registration(frame.registration_id).lock();
                let Some(pending) = registration.pending.get_mut(&frame.registration_id) else {
                    send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                    return Ok(());
                };
                match pending.bundle_id {
                    Some(existing) if existing != bundle_id => {
                        bail!(
                            "QUIC response registration {} changed bundle id from {} to {}",
                            frame.registration_id,
                            existing,
                            bundle_id
                        );
                    }
                    Some(_) => {}
                    None => pending.bundle_id = Some(bundle_id),
                }
                if pending.monitor_response.is_none() {
                    let (response_tx, response_rx) = oneshot::channel();
                    pending.monitor_response = Some(response_tx);
                    Some((
                        pending.context.clone(),
                        response_rx,
                        pending.monitor_cancel.clone(),
                    ))
                } else {
                    None
                }
            };
            // Bundle ownership is visible before the worker is allowed to call
            // generate(), so a pre-prologue bundle failure can resolve the
            // pending provider with an error.
            send_control(control_tx, FrameKind::Registered, frame.registration_id)?;
            if let Some((context, response_rx, monitor_cancel)) = monitor {
                spawn_response_monitor(
                    context,
                    response_rx,
                    frame.registration_id,
                    control_tx.clone(),
                    monitor_cancel,
                );
            }
        }
        FrameKind::Prologue => {
            if !frame.payload.is_empty() {
                bail!("QUIC response Prologue frame carried a payload");
            }
            let (response_tx, response_rx) = mpsc::channel(response_buffer_capacity);
            let (pending_connection, monitor_response) = {
                let mut registration = state.registration(frame.registration_id).lock();
                let Some(pending) = registration.pending.remove(&frame.registration_id) else {
                    send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                    return Ok(());
                };
                let PendingResponse {
                    context: _,
                    connection,
                    bundle_id: pending_bundle,
                    deferred,
                    monitor_cancel,
                    monitor_response,
                } = pending;
                if pending_bundle != Some(bundle_id) {
                    bail!(
                        "QUIC response registration {} was not registered to bundle {}",
                        frame.registration_id,
                        bundle_id
                    );
                }
                registration.active.insert(
                    frame.registration_id,
                    ActiveResponse {
                        sender: response_tx.clone(),
                        monitor_cancel: monitor_cancel.clone(),
                        bundle_id,
                        priority_ready: false,
                        priority_draining: false,
                        deferred,
                    },
                );
                (connection, monitor_response)
            };
            let Some(monitor_response) = monitor_response else {
                bail!(
                    "QUIC response registration {} had no cancellation monitor",
                    frame.registration_id
                );
            };
            let _ = monitor_response.send(response_tx.clone());
            if pending_connection
                .send(Ok(StreamReceiver { rx: response_rx }))
                .is_err()
            {
                remove_registration(state, frame.registration_id);
                send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                return Ok(());
            }
        }
        FrameKind::Error => {
            let error = String::from_utf8(frame.payload.to_vec())
                .context("QUIC response terminal error was not UTF-8")?;
            let pending = state
                .registration(frame.registration_id)
                .lock()
                .pending
                .remove(&frame.registration_id);
            if let Some(pending) = pending {
                if let Some(pending_bundle) = pending.bundle_id
                    && pending_bundle != bundle_id
                {
                    bail!(
                        "QUIC response registration {} changed bundle id from {} to {}",
                        frame.registration_id,
                        pending_bundle,
                        bundle_id
                    );
                }
                let _ = pending.connection.send(Err(error));
                remove_registration(state, frame.registration_id);
            } else {
                send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
            }
        }
        FrameKind::FirstData | FrameKind::PriorityEnd => {
            let first_payload = (frame.kind == FrameKind::FirstData).then_some(frame.payload);
            let active = {
                let mut registration = state.registration(frame.registration_id).lock();
                match registration.active.get_mut(&frame.registration_id) {
                    Some(active) if active.bundle_id != bundle_id => {
                        bail!(
                            "QUIC response registration {} changed bundle id from {} to {}",
                            frame.registration_id,
                            active.bundle_id,
                            bundle_id
                        );
                    }
                    Some(active) if active.priority_ready || active.priority_draining => None,
                    Some(active) => {
                        active.priority_draining = true;
                        Some(active.sender.clone())
                    }
                    None => None,
                }
            };
            let Some(sender) = active else {
                send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                return Ok(());
            };

            if let Some(payload) = first_payload {
                let delivery = deliver_server_payload(&sender, payload);
                if let Delivery::Reset(reason) = delivery {
                    reset_registration(state, control_tx, frame.registration_id, reason)?;
                    return Ok(());
                }
            }

            drain_deferred_response(state, frame.registration_id, &sender, control_tx).await?;
        }
        FrameKind::Data => {
            enum Disposition {
                Deferred,
                Deliver(mpsc::Sender<Bytes>, Bytes),
                Reset,
                Overflow,
            }

            let disposition = {
                let mut registration = state.registration(frame.registration_id).lock();
                match registration.active.get_mut(&frame.registration_id) {
                    Some(active) if active.bundle_id != bundle_id => {
                        bail!(
                            "QUIC response registration {} changed bundle id from {} to {}",
                            frame.registration_id,
                            active.bundle_id,
                            bundle_id
                        );
                    }
                    Some(active) if !active.priority_ready => {
                        match active.deferred.push(frame.payload) {
                            Ok(()) => Disposition::Deferred,
                            Err(_) => Disposition::Overflow,
                        }
                    }
                    Some(active) => Disposition::Deliver(active.sender.clone(), frame.payload),
                    None => {
                        if let Some(pending) = registration.pending.get_mut(&frame.registration_id)
                        {
                            if let Some(pending_bundle) = pending.bundle_id
                                && pending_bundle != bundle_id
                            {
                                bail!(
                                    "QUIC response registration {} changed bundle id from {} to {}",
                                    frame.registration_id,
                                    pending_bundle,
                                    bundle_id
                                );
                            }
                            pending.bundle_id = Some(bundle_id);
                            match pending.deferred.push(frame.payload) {
                                Ok(()) => Disposition::Deferred,
                                Err(_) => Disposition::Overflow,
                            }
                        } else {
                            Disposition::Reset
                        }
                    }
                }
            };
            match disposition {
                Disposition::Deferred => {}
                Disposition::Deliver(sender, payload) => {
                    if let Delivery::Reset(reason) = deliver_server_payload(&sender, payload) {
                        reset_registration(state, control_tx, frame.registration_id, reason)?;
                    }
                }
                Disposition::Reset => {
                    send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                }
                Disposition::Overflow => {
                    tracing::warn!(
                        registration_id = %frame.registration_id,
                        "QUIC response deferred-data bound exceeded"
                    );
                    remove_registration(state, frame.registration_id);
                    send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                }
            }
        }
        FrameKind::End => {
            if !frame.payload.is_empty() {
                bail!("QUIC response End frame carried a payload");
            }
            let disposition = {
                let mut registration = state.registration(frame.registration_id).lock();
                match registration.active.get_mut(&frame.registration_id) {
                    Some(active) if active.bundle_id != bundle_id => {
                        bail!(
                            "QUIC response registration {} changed bundle id from {} to {}",
                            frame.registration_id,
                            active.bundle_id,
                            bundle_id
                        );
                    }
                    Some(active) if !active.priority_ready => {
                        active.deferred.end = true;
                        0
                    }
                    Some(_) => 1,
                    None => {
                        if let Some(pending) = registration.pending.get_mut(&frame.registration_id)
                        {
                            if let Some(pending_bundle) = pending.bundle_id
                                && pending_bundle != bundle_id
                            {
                                bail!(
                                    "QUIC response registration {} changed bundle id from {} to {}",
                                    frame.registration_id,
                                    pending_bundle,
                                    bundle_id
                                );
                            }
                            pending.bundle_id = Some(bundle_id);
                            pending.deferred.end = true;
                            0
                        } else {
                            2
                        }
                    }
                }
            };
            match disposition {
                0 => {}
                1 => remove_registration(state, frame.registration_id),
                2 => {
                    send_control(control_tx, FrameKind::Reset, frame.registration_id)?;
                }
                _ => unreachable!(),
            }
        }
        FrameKind::Reset => {
            if !frame.payload.is_empty() {
                bail!("QUIC response Reset frame carried a payload");
            }
            let registration = state.registration(frame.registration_id).lock();
            let registered_bundle = registration
                .pending
                .get(&frame.registration_id)
                .and_then(|pending| pending.bundle_id)
                .or_else(|| {
                    registration
                        .active
                        .get(&frame.registration_id)
                        .map(|active| active.bundle_id)
                });
            if registered_bundle != Some(bundle_id) {
                bail!(
                    "QUIC response registration {} was not registered to bundle {}",
                    frame.registration_id,
                    bundle_id
                );
            }
            drop(registration);
            remove_registration(state, frame.registration_id);
        }
        FrameKind::Stop | FrameKind::Kill | FrameKind::Registered | FrameKind::Bundle => {
            bail!("worker sent reverse-only control frame {:?}", frame.kind)
        }
    }
    Ok(())
}

enum Delivery {
    Delivered,
    Reset(&'static str),
}

fn deliver_server_payload(sender: &mpsc::Sender<Bytes>, payload: Bytes) -> Delivery {
    match sender.try_send(payload) {
        Ok(()) => Delivery::Delivered,
        Err(mpsc::error::TrySendError::Full(_)) => Delivery::Reset("full"),
        Err(mpsc::error::TrySendError::Closed(_)) => Delivery::Reset("closed"),
    }
}

fn reset_registration(
    state: &ServerState,
    control_tx: &mpsc::Sender<Frame>,
    registration_id: Uuid,
    reason: &'static str,
) -> Result<()> {
    crate::metrics::quic_response::SERVER_MAILBOX_RESETS_TOTAL
        .with_label_values(&[reason])
        .inc();
    remove_registration(state, registration_id);
    send_control(control_tx, FrameKind::Reset, registration_id)
}

async fn drain_deferred_response(
    state: &Arc<ServerState>,
    registration_id: Uuid,
    sender: &mpsc::Sender<Bytes>,
    control_tx: &mpsc::Sender<Frame>,
) -> Result<()> {
    loop {
        let (deferred, remove_after_drain) = {
            let mut registration = state.registration(registration_id).lock();
            let Some(active) = registration.active.get_mut(&registration_id) else {
                return Ok(());
            };
            if active.deferred.data.is_empty() {
                active.priority_ready = true;
                active.priority_draining = false;
                (VecDeque::new(), active.deferred.end)
            } else {
                (active.deferred.take_data(), false)
            }
        };

        if deferred.is_empty() {
            if remove_after_drain {
                remove_registration(state, registration_id);
            }
            return Ok(());
        }

        for payload in deferred {
            if let Delivery::Reset(reason) = deliver_server_payload(sender, payload) {
                reset_registration(state, control_tx, registration_id, reason)?;
                return Ok(());
            }
        }
    }
}

fn spawn_response_monitor(
    context: Arc<dyn AsyncEngineContext>,
    response_rx: oneshot::Receiver<mpsc::Sender<Bytes>>,
    registration_id: Uuid,
    control_tx: mpsc::Sender<Frame>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let killed = context.killed();
        let stopped = context.stopped();
        tokio::pin!(killed, stopped, response_rx);
        let mut stop_sent = false;
        let response_tx = loop {
            let kind = tokio::select! {
                _ = cancel.cancelled() => return,
                _ = &mut killed => Some(FrameKind::Kill),
                _ = &mut stopped, if !stop_sent => Some(FrameKind::Stop),
                response = &mut response_rx => match response {
                    Ok(response_tx) => break response_tx,
                    Err(_) => return,
                },
            };
            let Some(kind) = kind else { unreachable!() };
            if send_control(&control_tx, kind, registration_id).is_err() {
                return;
            }
            if kind == FrameKind::Stop {
                stop_sent = true;
            } else {
                return;
            }
        };

        let closed = response_tx.closed();
        tokio::pin!(closed);
        loop {
            let kind = tokio::select! {
                _ = cancel.cancelled() => return,
                _ = &mut killed => FrameKind::Kill,
                _ = &mut stopped, if !stop_sent => FrameKind::Stop,
                _ = &mut closed => FrameKind::Reset,
            };
            if send_control(&control_tx, kind, registration_id).is_err() {
                return;
            }
            if kind == FrameKind::Stop {
                stop_sent = true;
            } else {
                return;
            }
        }
    });
}

fn send_control(
    control_tx: &mpsc::Sender<Frame>,
    kind: FrameKind,
    registration_id: Uuid,
) -> Result<()> {
    control_tx
        .try_send(Frame::new(kind, registration_id, Bytes::new()))
        .map_err(|error| anyhow!("QUIC response reverse-control enqueue failed: {error}"))
}

struct Lane {
    index: usize,
    sender: LaneSender,
}

struct ResponseLanes {
    ordered: Arc<Lane>,
    priority: Arc<Lane>,
}

struct ClientConnectionBundle {
    _endpoints: Arc<[quinn::Endpoint]>,
    connections: Arc<[quinn::Connection]>,
    lanes: Arc<[ResponseLanes]>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    healthy: Arc<AtomicBool>,
}

impl ClientConnectionBundle {
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConnectionKey {
    address: String,
    frontend_id: String,
    certificate_sha256: String,
}

struct ClientPoolEntry {
    current: ArcSwapOption<ClientConnectionBundle>,
    reconnect: tokio::sync::Mutex<()>,
}

impl ClientPoolEntry {
    fn new() -> Self {
        Self {
            current: ArcSwapOption::empty(),
            reconnect: tokio::sync::Mutex::new(()),
        }
    }
}

pub struct QuicResponseClientPool {
    config: QuicResponseConfig,
    connections: DashMap<ConnectionKey, Arc<ClientPoolEntry>>,
}

static PROCESS_CLIENT_POOL: OnceLock<Arc<QuicResponseClientPool>> = OnceLock::new();

pub fn process_client_pool_from_env() -> Result<Arc<QuicResponseClientPool>, PipelineError> {
    if let Some(pool) = PROCESS_CLIENT_POOL.get() {
        return Ok(pool.clone());
    }
    let pool = QuicResponseClientPool::from_env()?;
    Ok(PROCESS_CLIENT_POOL.get_or_init(|| pool).clone())
}

impl fmt::Debug for QuicResponseClientPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuicResponseClientPool")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl QuicResponseClientPool {
    pub fn from_env() -> Result<Arc<Self>, PipelineError> {
        Ok(Arc::new(Self {
            config: QuicResponseConfig::from_env()?,
            connections: DashMap::new(),
        }))
    }

    pub async fn sender(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ConnectionInfo,
    ) -> Result<QuicResponseSender> {
        self.sender_with_cancellation_metric(context, connection_info, None)
            .await
    }

    pub async fn sender_with_cancellation_metric(
        &self,
        context: Arc<dyn AsyncEngineContext>,
        connection_info: ConnectionInfo,
        cancellation_counter: Option<IntCounter>,
    ) -> Result<QuicResponseSender> {
        let info = QuicResponseConnectionInfo::try_from(connection_info)?;
        if info.request_id != context.id() {
            bail!(
                "QUIC response request id mismatch: connection has {}, context has {}",
                info.request_id,
                context.id()
            );
        }

        let key = ConnectionKey {
            address: info.address.clone(),
            frontend_id: info.frontend_id.clone(),
            certificate_sha256: info.certificate_sha256.clone(),
        };
        let connection = self.connection(key).await?;
        let lane_index = xxh3_64(info.request_id.as_bytes()) as usize % connection.lanes.len();
        let lanes = &connection.lanes[lane_index];
        let (registered_tx, registered_rx) = oneshot::channel();
        let response_context = Arc::new(ClientResponseContext {
            context: context.clone(),
            cancellation_counter,
            cancellation_recorded: AtomicBool::new(false),
            registered: Mutex::new(Some(registered_tx)),
        });
        connection
            .contexts
            .lock()
            .insert(info.registration_id, response_context.clone());
        let mut sender = QuicResponseSender {
            lane: lanes.ordered.clone(),
            priority_lane: lanes.priority.clone(),
            contexts: connection.contexts.clone(),
            response_context,
            registration_id: info.registration_id,
            prologue_sent: false,
            terminated: false,
            first_data_sent: AtomicBool::new(false),
        };

        if let Err(error) = sender
            .enqueue_on(
                &sender.priority_lane,
                Frame::new(FrameKind::Register, info.registration_id, Bytes::new()),
            )
            .await
        {
            connection.contexts.lock().remove(&info.registration_id);
            return Err(error);
        }

        // Register ownership before generate(), even if cancellation raced the
        // handshake. The registered response monitor then propagates that
        // cancellation, and generate() keeps its established cancellation
        // semantics instead of being skipped by transport setup.
        let registered = registered_rx
            .await
            .map_err(|_| anyhow!("QUIC response registration acknowledgement was dropped"))?;
        if let Err(error) = registered {
            if context.is_stopped() && !context.is_killed() {
                let _ = sender.abort().await;
            } else {
                connection.contexts.lock().remove(&info.registration_id);
                let _ = sender.priority_lane.sender.try_send(Frame::new(
                    FrameKind::Error,
                    info.registration_id,
                    Bytes::from(error.clone()),
                ));
            }
            bail!(error);
        }
        Ok(sender)
    }

    async fn connection(&self, key: ConnectionKey) -> Result<Arc<ClientConnectionBundle>> {
        let entry = self
            .connections
            .entry(key.clone())
            .or_insert_with(|| Arc::new(ClientPoolEntry::new()))
            .clone();
        if let Some(connection) = entry.current.load_full()
            && connection.is_healthy()
        {
            return Ok(connection);
        }

        let _reconnect = entry.reconnect.lock().await;
        if let Some(connection) = entry.current.load_full()
            && connection.is_healthy()
        {
            return Ok(connection);
        }
        let connection = self.connect(&key).await?;
        entry.current.store(Some(connection.clone()));
        Ok(connection)
    }

    async fn connect(&self, key: &ConnectionKey) -> Result<Arc<ClientConnectionBundle>> {
        let address: SocketAddr = key
            .address
            .parse()
            .with_context(|| format!("invalid QUIC response address {}", key.address))?;
        let expected_fingerprint = decode_fingerprint(&key.certificate_sha256)?;
        let client_config = pinned_client_config(expected_fingerprint)?;
        let priority_connection_index = BULK_CONNECTIONS;
        let total_connections = BULK_CONNECTIONS + PRIORITY_CONNECTIONS;
        let endpoints = self.client_endpoints(key, total_connections)?;
        let mut connections = Vec::with_capacity(total_connections);
        for endpoint in &endpoints {
            let connection = endpoint
                .connect_with(client_config.clone(), address, "localhost")?
                .await
                .with_context(|| {
                    format!("failed connecting QUIC response transport to {address}")
                })?;
            crate::metrics::quic_response::track_connection(connection.clone());
            connections.push(connection);
        }
        let connections: Arc<[quinn::Connection]> = connections.into();

        let bundle_id = Uuid::new_v4();
        let contexts = Arc::new(Mutex::new(HashMap::new()));
        let healthy = Arc::new(AtomicBool::new(true));
        let mut lanes = Vec::with_capacity(BULK_LANES);
        for index in 0..BULK_LANES {
            let connection = connections[index % BULK_CONNECTIONS].clone();
            let (send, recv) = connection.open_bi().await?;
            let (lane_tx, lane_rx) = lane_queue(BULK_LANE_QUEUE_CAPACITY);
            let ordered = Arc::new(Lane {
                index,
                sender: lane_tx,
            });
            spawn_client_lane(
                send,
                recv,
                lane_rx,
                self.config.batch_interval,
                MAX_BATCH_FRAMES,
                bundle_id,
                connections.clone(),
                contexts.clone(),
                healthy.clone(),
            );
            let priority_connection = connections[priority_connection_index].clone();
            let (priority_send, priority_recv) = priority_connection.open_bi().await?;
            let (priority_tx, priority_rx) = lane_queue(LANE_QUEUE_CAPACITY);
            let priority = Arc::new(Lane {
                index,
                sender: priority_tx,
            });
            spawn_client_lane(
                priority_send,
                priority_recv,
                priority_rx,
                Duration::ZERO,
                1,
                bundle_id,
                connections.clone(),
                contexts.clone(),
                healthy.clone(),
            );
            lanes.push(ResponseLanes { ordered, priority });
        }

        tracing::debug!(
            remote = %address,
            bulk_connections = BULK_CONNECTIONS,
            priority_connections = PRIORITY_CONNECTIONS,
            total_lanes = BULK_LANES,
            "QUIC response connection bundle and lanes ready"
        );
        Ok(Arc::new(ClientConnectionBundle {
            _endpoints: endpoints.into(),
            connections,
            lanes: lanes.into(),
            contexts,
            healthy,
        }))
    }

    fn client_endpoints(&self, key: &ConnectionKey, count: usize) -> Result<Vec<quinn::Endpoint>> {
        let ipv6 = key.address.parse::<SocketAddr>()?.is_ipv6();
        let mut endpoints = Vec::with_capacity(count);
        for _ in 0..count {
            let bind = if ipv6 {
                SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0)
            } else {
                SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
            };
            endpoints.push(quinn::Endpoint::client(bind)?);
        }
        Ok(endpoints)
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_client_lane(
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    receiver: LaneReceiver,
    interval: Duration,
    max_batch_frames: usize,
    bundle_id: Uuid,
    connections: Arc<[quinn::Connection]>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    healthy: Arc<AtomicBool>,
) {
    let writer_connections = connections.clone();
    let writer_contexts = contexts.clone();
    let writer_healthy = healthy.clone();
    tokio::spawn(async move {
        if let Err(error) = run_client_writer(
            send,
            receiver,
            interval,
            max_batch_frames,
            bundle_id,
            writer_contexts.clone(),
        )
        .await
        {
            fail_client_connection_bundle(
                &writer_connections,
                &writer_contexts,
                &writer_healthy,
                &error.to_string(),
            );
        }
    });

    tokio::spawn(async move {
        if let Err(error) = run_client_control_reader(recv, contexts.clone()).await {
            fail_client_connection_bundle(&connections, &contexts, &healthy, &error.to_string());
        }
    });
}

async fn run_client_writer(
    mut send: quinn::SendStream,
    mut receiver: LaneReceiver,
    interval: Duration,
    max_batch_frames: usize,
    bundle_id: Uuid,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
) -> Result<()> {
    let bundle = Frame::new(FrameKind::Bundle, bundle_id, Bytes::new());
    let mut bundle_chunks = [bundle.header()];
    send.write_all_chunks(&mut bundle_chunks).await?;

    let mut batch = Vec::with_capacity(max_batch_frames);
    let mut chunks = Vec::with_capacity(max_batch_frames * 2);
    while receive_batch(&mut receiver, &mut batch, interval, max_batch_frames).await {
        chunks.clear();
        for frame in &batch {
            chunks.push(frame.header());
            if !frame.payload.is_empty() {
                chunks.push(frame.payload.clone());
            }
        }
        send.write_all_chunks(&mut chunks).await?;
        if batch.iter().any(|frame| {
            matches!(
                frame.kind,
                FrameKind::Error | FrameKind::End | FrameKind::Reset
            )
        }) {
            let mut contexts = contexts.lock();
            for frame in &batch {
                if matches!(
                    frame.kind,
                    FrameKind::Error | FrameKind::End | FrameKind::Reset
                ) {
                    contexts.remove(&frame.registration_id);
                }
            }
        }
        batch.clear();
    }
    bail!("QUIC response lane queue closed unexpectedly")
}

async fn run_client_control_reader(
    recv: quinn::RecvStream,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
) -> Result<()> {
    let mut recv = BufReader::with_capacity(RECEIVE_BUFFER_CAPACITY, recv);
    loop {
        let frame = read_frame(&mut recv).await?;
        if !frame.payload.is_empty() {
            bail!("QUIC response reverse control carried a payload");
        }
        let entry = contexts.lock().get(&frame.registration_id).cloned();
        match frame.kind {
            FrameKind::Registered => {
                if let Some(entry) = entry
                    && let Some(registered) = entry.registered.lock().take()
                {
                    let _ = registered.send(Ok(()));
                }
            }
            FrameKind::Stop => {
                if let Some(entry) = entry {
                    entry.fail_registration("QUIC response registration was stopped");
                    entry.record_cancellation();
                    entry.context.stop();
                }
            }
            FrameKind::Kill | FrameKind::Reset => {
                if let Some(entry) = entry {
                    entry.fail_registration("QUIC response registration was reset");
                    entry.record_cancellation();
                    entry.context.kill();
                }
            }
            _ => bail!("frontend sent forward-only frame {:?}", frame.kind),
        }
    }
}

fn fail_client_connection_bundle(
    connections: &[quinn::Connection],
    contexts: &Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>,
    healthy: &AtomicBool,
    reason: &str,
) {
    if !healthy.swap(false, Ordering::AcqRel) {
        return;
    }
    crate::metrics::quic_response::record_bundle_failure("worker");
    tracing::warn!(%reason, "QUIC response connection bundle invariant failed");
    for (_, entry) in contexts.lock().drain() {
        entry.fail_registration(reason);
        entry.record_cancellation();
        entry.context.kill();
    }
    for connection in connections {
        connection.close(
            CLOSE_CODE_INVARIANT,
            b"response connection bundle invariant failure",
        );
    }
}

pub struct QuicResponseSender {
    lane: Arc<Lane>,
    priority_lane: Arc<Lane>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    response_context: Arc<ClientResponseContext>,
    registration_id: Uuid,
    prologue_sent: bool,
    terminated: bool,
    first_data_sent: AtomicBool,
}

struct ClientResponseContext {
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
    cancellation_recorded: AtomicBool,
    registered: Mutex<Option<oneshot::Sender<std::result::Result<(), String>>>>,
}

impl ClientResponseContext {
    fn fail_registration(&self, reason: &str) {
        if let Some(registered) = self.registered.lock().take() {
            let _ = registered.send(Err(reason.to_string()));
        }
    }

    fn record_cancellation(&self) {
        if !self.cancellation_recorded.swap(true, Ordering::AcqRel)
            && let Some(counter) = &self.cancellation_counter
        {
            counter.inc();
        }
    }
}

impl QuicResponseSender {
    #[cfg(test)]
    fn lane_index(&self) -> usize {
        self.lane.index
    }

    pub async fn send_prologue(&mut self, error: Option<String>) -> Result<(), String> {
        if self.prologue_sent {
            return Err("QUIC response prologue already sent".to_string());
        }
        let (kind, payload, terminal) = match error {
            Some(error) => (FrameKind::Error, Bytes::from(error), true),
            None => (FrameKind::Prologue, Bytes::new(), false),
        };
        self.enqueue_on(
            &self.priority_lane,
            Frame::new(kind, self.registration_id, payload),
        )
        .await
        .map_err(|error| error.to_string())?;
        self.prologue_sent = true;
        if terminal {
            self.terminated = true;
        }
        Ok(())
    }

    pub async fn send(&self, payload: Bytes) -> Result<()> {
        if !self.prologue_sent || self.terminated {
            bail!("QUIC response sender is not open for data");
        }
        let first_data = !self.first_data_sent.swap(true, Ordering::AcqRel);
        if first_data
            && self.response_context.context.is_stopped()
            && !self.response_context.context.is_killed()
        {
            bail!("QUIC response context stopped before its first response");
        }
        let frame = if first_data {
            Frame::new(FrameKind::FirstData, self.registration_id, payload)
        } else {
            Frame::new(FrameKind::Data, self.registration_id, payload)
        };
        if first_data {
            self.enqueue_on(&self.priority_lane, frame).await
        } else {
            self.enqueue_on(&self.lane, frame).await
        }
    }

    pub async fn finish(&mut self) -> Result<()> {
        if !self.prologue_sent || self.terminated {
            return Ok(());
        }
        if !self.first_data_sent.load(Ordering::Acquire) {
            self.enqueue_on(
                &self.priority_lane,
                Frame::new(FrameKind::PriorityEnd, self.registration_id, Bytes::new()),
            )
            .await?;
        }
        self.enqueue_on(
            &self.lane,
            Frame::new(FrameKind::End, self.registration_id, Bytes::new()),
        )
        .await?;
        self.terminated = true;
        Ok(())
    }

    /// Close one cancelled logical response without ending its shared lane.
    pub async fn abort(&mut self) -> Result<()> {
        if self.terminated {
            return Ok(());
        }
        if !self.prologue_sent {
            self.enqueue_on(
                &self.priority_lane,
                Frame::new(FrameKind::Prologue, self.registration_id, Bytes::new()),
            )
            .await?;
            self.prologue_sent = true;
        }
        self.response_context.record_cancellation();
        self.enqueue_on(
            &self.priority_lane,
            Frame::new(FrameKind::Reset, self.registration_id, Bytes::new()),
        )
        .await?;
        self.terminated = true;
        Ok(())
    }

    async fn enqueue_on(&self, lane: &Lane, frame: Frame) -> Result<()> {
        if frame.payload.len() > MAX_FRAME_PAYLOAD {
            bail!(
                "QUIC response frame payload {} exceeds {}",
                frame.payload.len(),
                MAX_FRAME_PAYLOAD
            );
        }
        match lane.sender.try_send(frame) {
            Ok(()) => Ok(()),
            Err(LaneTrySendError::Closed(_)) => {
                bail!("QUIC response lane closed")
            }
            Err(LaneTrySendError::Full(frame)) => lane.sender.send(frame).await,
        }
    }
}

impl Drop for QuicResponseSender {
    fn drop(&mut self) {
        if !self.prologue_sent {
            self.contexts.lock().remove(&self.registration_id);
            return;
        }
        if self.terminated {
            return;
        }
        let sender = self.lane.sender.clone();
        let priority_sender = (!self.first_data_sent.load(Ordering::Acquire))
            .then(|| self.priority_lane.sender.clone());
        let registration_id = self.registration_id;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                if let Some(priority_sender) = priority_sender {
                    let _ = priority_sender
                        .send(Frame::new(
                            FrameKind::PriorityEnd,
                            registration_id,
                            Bytes::new(),
                        ))
                        .await;
                }
                let _ = sender
                    .send(Frame::new(FrameKind::End, registration_id, Bytes::new()))
                    .await;
            });
        }
    }
}

#[derive(Debug)]
struct PinnedCertificateVerifier {
    expected: [u8; 32],
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl ServerCertVerifier for PinnedCertificateVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        let actual: [u8; 32] = Sha256::digest(end_entity.as_ref()).into();
        if actual != self.expected {
            return Err(rustls::Error::General(
                "QUIC response certificate fingerprint mismatch".to_string(),
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn pinned_client_config(expected: [u8; 32]) -> Result<quinn::ClientConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = Arc::new(PinnedCertificateVerifier {
        expected,
        provider: provider.clone(),
    });
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let crypto = QuicClientConfig::try_from(crypto)?;
    Ok(quinn::ClientConfig::new(Arc::new(crypto)))
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn decode_fingerprint(encoded: &str) -> Result<[u8; 32]> {
    let bytes = encoded.as_bytes();
    if bytes.len() != 64 || !bytes.is_ascii() {
        bail!("invalid ASCII SHA-256 fingerprint length {}", bytes.len());
    }
    let mut decoded = [0_u8; 32];
    for (pair, output) in bytes.chunks_exact(2).zip(&mut decoded) {
        let high = decode_hex_digit(pair[0])?;
        let low = decode_hex_digit(pair[1])?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

fn decode_hex_digit(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => bail!("invalid SHA-256 fingerprint hex digit"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::AsyncEngineContextProvider, pipeline::Context as PipelineContext};

    fn frame(index: usize) -> Frame {
        Frame::new(FrameKind::Data, Uuid::nil(), Bytes::from(index.to_string()))
    }

    fn test_pool() -> QuicResponseClientPool {
        QuicResponseClientPool {
            config: QuicResponseConfig {
                batch_interval: Duration::ZERO,
                response_buffer_capacity: RESPONSE_BUFFER_CAPACITY,
            },
            connections: DashMap::new(),
        }
    }

    fn test_server(
        response_buffer_capacity: usize,
    ) -> (CancellationToken, Arc<QuicResponseServer>) {
        let shutdown = CancellationToken::new();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = QuicResponseServer::new_with_config(
            address,
            address,
            shutdown.clone(),
            None,
            QuicResponseConfig {
                batch_interval: Duration::ZERO,
                response_buffer_capacity,
            },
        )
        .unwrap();
        (shutdown, server)
    }

    fn only_bundle(pool: &QuicResponseClientPool) -> Arc<ClientConnectionBundle> {
        assert_eq!(pool.connections.len(), 1);
        pool.connections
            .iter()
            .next()
            .unwrap()
            .current
            .load_full()
            .unwrap()
    }

    #[tokio::test]
    async fn fixed_batch_cap_drains_512_frames() {
        let (tx, mut rx) = lane_queue(MAX_BATCH_FRAMES + 1);
        for index in 0..=MAX_BATCH_FRAMES {
            tx.send(frame(index)).await.unwrap();
        }
        let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
        assert!(receive_batch(&mut rx, &mut batch, Duration::ZERO, MAX_BATCH_FRAMES).await);
        assert_eq!(batch.len(), MAX_BATCH_FRAMES);
        assert_eq!(batch[0].payload, Bytes::from_static(b"0"));
        assert_eq!(
            batch[MAX_BATCH_FRAMES - 1].payload,
            Bytes::from_static(b"511")
        );
    }

    #[tokio::test]
    async fn lane_queue_preserves_capacity_order_and_close() {
        let (tx, mut rx) = lane_queue(LANE_QUEUE_CAPACITY);
        let queue = tx.0.clone();
        for index in 0..LANE_QUEUE_CAPACITY {
            tx.try_send(frame(index)).unwrap();
        }
        assert_eq!(queue.notifications.load(Ordering::Relaxed), 1);
        assert!(matches!(
            tx.try_send(frame(LANE_QUEUE_CAPACITY)),
            Err(LaneTrySendError::Full(_))
        ));
        let blocked = tokio::spawn({
            let tx = tx.clone();
            async move { tx.send(frame(LANE_QUEUE_CAPACITY)).await }
        });
        tokio::task::yield_now().await;
        assert!(!blocked.is_finished());

        let mut batch = Vec::with_capacity(256);
        assert!(receive_batch(&mut rx, &mut batch, Duration::ZERO, 256).await);
        for (expected, actual) in (0..256).zip(&batch) {
            assert_eq!(actual.payload, Bytes::from(expected.to_string()));
        }

        blocked.await.unwrap().unwrap();
        assert_eq!(queue.notifications.load(Ordering::Relaxed), 1);
        assert!(receive_batch(&mut rx, &mut batch, Duration::ZERO, LANE_QUEUE_CAPACITY).await);
        for (expected, actual) in (256..=LANE_QUEUE_CAPACITY).zip(&batch) {
            assert_eq!(actual.payload, Bytes::from(expected.to_string()));
        }

        tx.try_send(frame(LANE_QUEUE_CAPACITY + 1)).unwrap();
        assert_eq!(queue.notifications.load(Ordering::Relaxed), 1);
        drop(tx);
        assert!(receive_batch(&mut rx, &mut batch, Duration::ZERO, LANE_QUEUE_CAPACITY).await);
        assert_eq!(batch.len(), 1);
        assert_eq!(
            batch[0].payload,
            Bytes::from((LANE_QUEUE_CAPACITY + 1).to_string())
        );
        assert!(!receive_batch(&mut rx, &mut batch, Duration::ZERO, LANE_QUEUE_CAPACITY).await);
    }

    #[tokio::test]
    async fn lane_queue_close_has_no_lost_wakeup() {
        let (tx, mut rx) = lane_queue(1);
        let waiting = tokio::spawn(async move {
            let mut batch = Vec::new();
            rx.recv_many(&mut batch, 1).await
        });
        tokio::task::yield_now().await;
        drop(tx);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), waiting)
                .await
                .expect("closed queue must wake its receiver")
                .unwrap(),
            0
        );
    }

    #[tokio::test(start_paused = true)]
    async fn frames_arriving_before_deadline_join_the_same_batch() {
        let (tx, mut rx) = lane_queue(128);
        tx.send(frame(0)).await.unwrap();
        let task = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
            let wait = receive_batch(
                &mut rx,
                &mut batch,
                Duration::from_millis(1),
                MAX_BATCH_FRAMES,
            )
            .await;
            (wait, batch.len())
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_micros(500)).await;
        for index in 1..64 {
            tx.send(frame(index)).await.unwrap();
        }
        tokio::task::yield_now().await;
        let (wait, len) = task.await.unwrap();
        assert!(wait);
        assert_eq!(len, 64);
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_wins_when_a_frame_arrives_at_the_boundary() {
        let (tx, mut rx) = lane_queue(128);
        tx.send(frame(0)).await.unwrap();
        let task = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
            let wait = receive_batch(
                &mut rx,
                &mut batch,
                Duration::from_millis(1),
                MAX_BATCH_FRAMES,
            )
            .await;
            (wait, batch, rx)
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(1)).await;
        tx.send(frame(1)).await.unwrap();
        tokio::task::yield_now().await;

        let (wait, batch, mut rx) = task.await.unwrap();
        assert!(wait);
        assert_eq!(batch.len(), 1);
        assert_eq!(rx.recv().await.unwrap().payload, Bytes::from_static(b"1"));
    }

    #[tokio::test]
    async fn eight_connections_eight_lanes_carry_1000_ordered_responses() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();

        let mut lane_by_request = HashMap::new();
        for index in 0..1_000 {
            let context = PipelineContext::new(());
            let request_id = context.id().to_string();
            let registered = server.register_response(context.context());
            let (connection_info, provider) = registered.into_parts();
            let mut sender = pool
                .sender(context.context(), connection_info)
                .await
                .unwrap();
            let expected_lane = xxh3_64(request_id.as_bytes()) as usize % 8;
            assert_eq!(sender.lane_index(), expected_lane);
            lane_by_request.insert(request_id, sender.lane_index());
            sender.send_prologue(None).await.unwrap();
            sender
                .send(Bytes::from(format!("{index}:first")))
                .await
                .unwrap();
            sender
                .send(Bytes::from(format!("{index}:second")))
                .await
                .unwrap();
            sender.finish().await.unwrap();

            let mut receiver = provider.await.unwrap().unwrap();
            assert_eq!(
                receiver.rx.recv().await.unwrap(),
                Bytes::from(format!("{index}:first"))
            );
            assert_eq!(
                receiver.rx.recv().await.unwrap(),
                Bytes::from(format!("{index}:second"))
            );
            assert!(receiver.rx.recv().await.is_none());
        }

        assert_eq!(lane_by_request.len(), 1_000);
        let bundle = only_bundle(&pool);
        assert_eq!(bundle.connections.len(), 9); // Eight bulk and one priority.
        assert_eq!(bundle.lanes.len(), 8);
        shutdown.cancel();
    }

    #[tokio::test]
    async fn registration_ack_does_not_resolve_response_stream() {
        let (shutdown, server) = test_server(4);
        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let registration_id = registered.registration_id().unwrap();
        let (_, mut provider) = registered.into_parts();
        let bundle_id = Uuid::new_v4();
        let (control_tx, mut control_rx) = mpsc::channel(4);

        process_server_frame(
            Frame::new(FrameKind::Register, registration_id, Bytes::new()),
            bundle_id,
            &server.state,
            &control_tx,
            4,
        )
        .await
        .unwrap();
        assert_eq!(control_rx.recv().await.unwrap().kind, FrameKind::Registered);
        assert!(matches!(
            provider.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        process_server_frame(
            Frame::new(FrameKind::Prologue, registration_id, Bytes::new()),
            bundle_id,
            &server.state,
            &control_tx,
            4,
        )
        .await
        .unwrap();
        assert!(provider.await.unwrap().is_ok());
        shutdown.cancel();
    }

    #[tokio::test]
    async fn prologue_activation_is_atomic_with_concurrent_bulk_data() {
        let (shutdown, server) = test_server(4);
        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let registration_id = registered.registration_id().unwrap();
        let (_, provider) = registered.into_parts();
        let bundle_id = Uuid::new_v4();
        let (control_tx, mut control_rx) = mpsc::channel(8);

        process_server_frame(
            Frame::new(FrameKind::Register, registration_id, Bytes::new()),
            bundle_id,
            &server.state,
            &control_tx,
            4,
        )
        .await
        .unwrap();
        assert_eq!(control_rx.recv().await.unwrap().kind, FrameKind::Registered);

        let (prologue, bulk) = tokio::join!(
            process_server_frame(
                Frame::new(FrameKind::Prologue, registration_id, Bytes::new()),
                bundle_id,
                &server.state,
                &control_tx,
                4,
            ),
            process_server_frame(
                Frame::new(
                    FrameKind::Data,
                    registration_id,
                    Bytes::from_static(b"bulk"),
                ),
                bundle_id,
                &server.state,
                &control_tx,
                4,
            )
        );
        prologue.unwrap();
        bulk.unwrap();

        process_server_frame(
            Frame::new(
                FrameKind::FirstData,
                registration_id,
                Bytes::from_static(b"first"),
            ),
            bundle_id,
            &server.state,
            &control_tx,
            4,
        )
        .await
        .unwrap();

        let mut receiver = provider.await.unwrap().unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"first")
        );
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"bulk")
        );
        shutdown.cancel();
    }

    #[tokio::test]
    async fn priority_path_preserves_response_order() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();

        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let (connection_info, provider) = registered.into_parts();
        let mut sender = pool
            .sender(context.context(), connection_info)
            .await
            .unwrap();
        sender.send_prologue(None).await.unwrap();
        sender.send(Bytes::from_static(b"first")).await.unwrap();
        sender.send(Bytes::from_static(b"second")).await.unwrap();
        sender.finish().await.unwrap();

        let mut receiver = provider.await.unwrap().unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"first")
        );
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"second")
        );
        assert!(receiver.rx.recv().await.is_none());
        shutdown.cancel();
    }

    #[tokio::test]
    async fn receiver_drop_sends_logical_reset_without_resetting_lane() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();
        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let (info, provider) = registered.into_parts();
        let mut sender = pool.sender(context.context(), info).await.unwrap();
        sender.send_prologue(None).await.unwrap();
        let receiver = provider.await.unwrap().unwrap();
        drop(receiver);

        let worker_context = context.context();
        tokio::time::timeout(Duration::from_secs(1), worker_context.killed())
            .await
            .expect("logical reset should kill the matching worker context");
        assert!(only_bundle(&pool).is_healthy());

        // A sibling response still uses the same healthy connection.
        let sibling = PipelineContext::new(());
        let registered = server.register_response(sibling.context());
        let (info, provider) = registered.into_parts();
        let mut sibling_sender = pool.sender(sibling.context(), info).await.unwrap();
        sibling_sender.send_prologue(None).await.unwrap();
        sibling_sender.finish().await.unwrap();
        let mut receiver = provider.await.unwrap().unwrap();
        assert!(receiver.rx.recv().await.is_none());
        shutdown.cancel();
    }

    #[tokio::test]
    async fn priority_barrier_reorders_early_bulk_frames() {
        let registration_id = Uuid::new_v4();
        let bundle_id = Uuid::new_v4();
        let (response_tx, mut response_rx) = mpsc::channel(4);
        let state = Arc::new(ServerState::default());
        state.registration(registration_id).lock().active.insert(
            registration_id,
            ActiveResponse {
                sender: response_tx,
                monitor_cancel: CancellationToken::new(),
                bundle_id,
                priority_ready: false,
                priority_draining: false,
                deferred: DeferredResponse::default(),
            },
        );
        let (control_tx, mut control_rx) = mpsc::channel(4);

        process_server_frame(
            Frame::new(
                FrameKind::Data,
                registration_id,
                Bytes::from_static(b"second"),
            ),
            bundle_id,
            &state,
            &control_tx,
            4,
        )
        .await
        .unwrap();
        process_server_frame(
            Frame::new(FrameKind::End, registration_id, Bytes::new()),
            bundle_id,
            &state,
            &control_tx,
            4,
        )
        .await
        .unwrap();
        assert!(matches!(
            response_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        process_server_frame(
            Frame::new(
                FrameKind::FirstData,
                registration_id,
                Bytes::from_static(b"first"),
            ),
            bundle_id,
            &state,
            &control_tx,
            4,
        )
        .await
        .unwrap();

        assert_eq!(
            response_rx.recv().await.unwrap(),
            Bytes::from_static(b"first")
        );
        assert_eq!(
            response_rx.recv().await.unwrap(),
            Bytes::from_static(b"second")
        );
        assert!(response_rx.recv().await.is_none());
        assert!(matches!(
            control_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[test]
    fn deferred_response_enforces_frame_and_byte_bounds() {
        let mut frame_bounded = DeferredResponse::default();
        for _ in 0..MAX_DEFERRED_RESPONSE_FRAMES {
            frame_bounded.push(Bytes::new()).unwrap();
        }
        assert!(frame_bounded.push(Bytes::from_static(b"overflow")).is_err());

        let mut byte_bounded = DeferredResponse::default();
        byte_bounded
            .push(Bytes::from(vec![0; MAX_DEFERRED_RESPONSE_BYTES]))
            .unwrap();
        assert!(byte_bounded.push(Bytes::from_static(b"overflow")).is_err());
    }

    #[tokio::test]
    async fn terminal_generate_error_uses_ordered_lane_frame() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();
        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let (info, provider) = registered.into_parts();
        let mut sender = pool.sender(context.context(), info).await.unwrap();
        sender
            .send_prologue(Some("generate failed".to_string()))
            .await
            .unwrap();
        match provider.await.unwrap() {
            Err(error) => assert_eq!(error, "generate failed"),
            Ok(_) => panic!("terminal error unexpectedly opened a response stream"),
        }
        shutdown.cancel();
    }

    #[tokio::test]
    async fn cancelled_sender_closes_only_its_logical_response() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();

        let cancelled = PipelineContext::new(());
        let registered = server.register_response(cancelled.context());
        let (info, provider) = registered.into_parts();
        let mut sender = pool.sender(cancelled.context(), info).await.unwrap();
        sender.abort().await.unwrap();
        let mut receiver = provider.await.unwrap().unwrap();
        assert!(receiver.rx.recv().await.is_none());

        let sibling = PipelineContext::new(());
        let registered = server.register_response(sibling.context());
        let (info, provider) = registered.into_parts();
        let mut sibling_sender = pool.sender(sibling.context(), info).await.unwrap();
        sibling_sender.send_prologue(None).await.unwrap();
        sibling_sender.finish().await.unwrap();
        let mut receiver = provider.await.unwrap().unwrap();
        assert!(receiver.rx.recv().await.is_none());
        assert!(only_bundle(&pool).is_healthy());
        shutdown.cancel();
    }

    #[tokio::test]
    async fn physical_connection_failure_kills_all_and_reconnects_once() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = Arc::new(test_pool());

        let first_context = PipelineContext::new(());
        let first = server.register_response(first_context.context());
        let first_info = first.connection_info.clone();
        let mut first_sender = pool
            .sender(first_context.context(), first_info.clone())
            .await
            .unwrap();
        first_sender.send_prologue(None).await.unwrap();
        let old_connections = only_bundle(&pool)
            .connections
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let old_ids = old_connections
            .iter()
            .map(quinn::Connection::stable_id)
            .collect::<Vec<_>>();
        old_connections[0].close(CLOSE_CODE_INVARIANT, b"test lane failure");
        let first_engine_context = first_context.context();
        tokio::time::timeout(Duration::from_secs(1), first_engine_context.killed())
            .await
            .expect("connection failure should kill every active context");
        tokio::time::timeout(Duration::from_secs(1), async {
            while old_connections
                .iter()
                .any(|connection| connection.close_reason().is_none())
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("one physical failure must close the complete client bundle");
        let mut registrations = Vec::new();
        let mut providers = Vec::new();
        for _ in 0..16 {
            let context = PipelineContext::new(());
            let registered = server.register_response(context.context());
            let (connection_info, provider) = registered.into_parts();
            registrations.push((context, connection_info));
            providers.push(provider);
        }
        let senders =
            futures::future::join_all(registrations.into_iter().map(|(context, info)| {
                let pool = pool.clone();
                async move { pool.sender(context.context(), info).await }
            }))
            .await;
        let errors = senders
            .iter()
            .filter_map(|result| result.as_ref().err().map(ToString::to_string))
            .collect::<Vec<_>>();
        assert!(errors.is_empty(), "replacement sender errors: {errors:?}");
        let replacement = only_bundle(&pool);
        assert_eq!(
            replacement.connections.len(),
            BULK_CONNECTIONS + PRIORITY_CONNECTIONS
        );
        assert!(
            replacement
                .connections
                .iter()
                .all(|connection| !old_ids.contains(&connection.stable_id()))
        );
        assert_eq!(replacement.lanes.len(), 8);
        drop(providers);
        shutdown.cancel();
    }

    #[tokio::test]
    async fn different_connection_keys_initialize_in_parallel() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();

        let blocked_key = ConnectionKey {
            address: "127.0.0.1:1".to_string(),
            frontend_id: "blocked".to_string(),
            certificate_sha256: "00".repeat(32),
        };
        let blocked_entry = pool
            .connections
            .entry(blocked_key)
            .or_insert_with(|| Arc::new(ClientPoolEntry::new()))
            .clone();
        let blocked_reconnect = blocked_entry.reconnect.lock().await;

        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let (info, provider) = registered.into_parts();
        let sender =
            tokio::time::timeout(Duration::from_secs(1), pool.sender(context.context(), info))
                .await
                .expect("one frontend key must not wait for another key's reconnect mutex")
                .unwrap();

        drop(sender);
        drop(provider);
        drop(blocked_reconnect);
        shutdown.cancel();
    }

    #[tokio::test]
    async fn bundle_failure_before_prologue_resolves_pending_provider() {
        let (shutdown, server) = test_server(RESPONSE_BUFFER_CAPACITY);
        let pool = test_pool();
        let context = PipelineContext::new(());
        let registered = server.register_response(context.context());
        let (info, provider) = registered.into_parts();
        let _sender = pool.sender(context.context(), info).await.unwrap();
        let bundle = only_bundle(&pool);
        bundle.connections[0].close(CLOSE_CODE_INVARIANT, b"test pre-prologue failure");

        let result = tokio::time::timeout(Duration::from_secs(2), provider)
            .await
            .expect("pending provider must resolve after bundle failure")
            .unwrap();
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("pre-prologue bundle failure unexpectedly opened a stream"),
        };
        assert!(error.contains("bundle failed"));
        tokio::time::timeout(Duration::from_secs(2), context.context().killed())
            .await
            .expect("bundle failure must kill the worker request context");
        shutdown.cancel();
    }

    #[tokio::test]
    async fn full_mailbox_resets_only_its_registration() {
        let state = Arc::new(ServerState::default());
        let bundle_id = Uuid::new_v4();
        let full_id = Uuid::new_v4();
        let sibling_id = Uuid::new_v4();
        let (full_tx, mut full_rx) = mpsc::channel(1);
        full_tx.try_send(Bytes::from_static(b"occupied")).unwrap();
        let (sibling_tx, mut sibling_rx) = mpsc::channel(1);
        for (registration_id, sender) in [(full_id, full_tx), (sibling_id, sibling_tx)] {
            state.registration(registration_id).lock().active.insert(
                registration_id,
                ActiveResponse {
                    sender,
                    monitor_cancel: CancellationToken::new(),
                    bundle_id,
                    priority_ready: true,
                    priority_draining: false,
                    deferred: DeferredResponse::default(),
                },
            );
        }
        let instance = EndpointInstanceId {
            namespace: "n".to_string(),
            component: "c".to_string(),
            endpoint: "e".to_string(),
            instance_id: 1,
        };
        {
            let mut indexes = state.indexes.lock();
            indexes
                .registration_instance
                .insert(full_id, instance.clone());
            indexes
                .instance_registrations
                .insert(instance.clone(), vec![full_id]);
        }
        let (control_tx, mut control_rx) = mpsc::channel(4);

        process_server_frame(
            Frame::new(FrameKind::Data, full_id, Bytes::from_static(b"reset")),
            bundle_id,
            &state,
            &control_tx,
            1,
        )
        .await
        .unwrap();
        assert_eq!(control_rx.recv().await.unwrap().kind, FrameKind::Reset);
        assert!(
            !state
                .registration(full_id)
                .lock()
                .active
                .contains_key(&full_id)
        );
        assert!(
            !state
                .indexes
                .lock()
                .registration_instance
                .contains_key(&full_id)
        );
        assert_eq!(
            full_rx.recv().await.unwrap(),
            Bytes::from_static(b"occupied")
        );

        process_server_frame(
            Frame::new(FrameKind::Data, sibling_id, Bytes::from_static(b"sibling")),
            bundle_id,
            &state,
            &control_tx,
            1,
        )
        .await
        .unwrap();
        assert_eq!(
            sibling_rx.recv().await.unwrap(),
            Bytes::from_static(b"sibling")
        );
    }

    #[tokio::test]
    async fn priority_delivery_failures_remove_all_registration_indexes() {
        for deferred_failure in [false, true] {
            let state = Arc::new(ServerState::default());
            let bundle_id = Uuid::new_v4();
            let registration_id = Uuid::new_v4();
            let (response_tx, _response_rx) = mpsc::channel(1);
            let mut deferred = DeferredResponse::default();
            if deferred_failure {
                deferred.push(Bytes::from_static(b"bulk")).unwrap();
            } else {
                response_tx
                    .try_send(Bytes::from_static(b"occupied"))
                    .unwrap();
            }
            state.registration(registration_id).lock().active.insert(
                registration_id,
                ActiveResponse {
                    sender: response_tx,
                    monitor_cancel: CancellationToken::new(),
                    bundle_id,
                    priority_ready: false,
                    priority_draining: false,
                    deferred,
                },
            );
            let instance = EndpointInstanceId {
                namespace: "n".to_string(),
                component: "c".to_string(),
                endpoint: "e".to_string(),
                instance_id: registration_id.as_u128() as u64,
            };
            {
                let mut indexes = state.indexes.lock();
                indexes
                    .registration_instance
                    .insert(registration_id, instance.clone());
                indexes
                    .instance_registrations
                    .insert(instance.clone(), vec![registration_id]);
            }
            let (control_tx, mut control_rx) = mpsc::channel(4);

            process_server_frame(
                Frame::new(
                    FrameKind::FirstData,
                    registration_id,
                    Bytes::from_static(b"first"),
                ),
                bundle_id,
                &state,
                &control_tx,
                1,
            )
            .await
            .unwrap();

            assert_eq!(control_rx.recv().await.unwrap().kind, FrameKind::Reset);
            assert!(
                !state
                    .registration(registration_id)
                    .lock()
                    .active
                    .contains_key(&registration_id)
            );
            let indexes = state.indexes.lock();
            assert!(!indexes.registration_instance.contains_key(&registration_id));
            assert!(!indexes.instance_registrations.contains_key(&instance));
        }
    }

    #[test]
    fn fingerprint_round_trip() {
        let fingerprint = [0x5a; 32];
        assert_eq!(
            decode_fingerprint(&encode_hex(&fingerprint)).unwrap(),
            fingerprint
        );
        assert!(decode_fingerprint(&"é".repeat(32)).is_err());
    }

    #[tokio::test]
    async fn server_drop_cancels_accept_loops_and_closes_endpoints() {
        let (shutdown, server) = test_server(1);
        assert_eq!(server.endpoints.len(), SERVER_ENDPOINTS);
        let accept_shutdown = server.shutdown.clone();
        drop(server);
        assert!(accept_shutdown.is_cancelled());
        assert!(!shutdown.is_cancelled());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn primary_udp_bind_is_exclusive_before_reuseport_siblings_join() {
        let ephemeral = "127.0.0.1:0".parse().unwrap();
        let primary = bind_server_udp(ephemeral, false).unwrap();
        let bound = primary.local_addr().unwrap();
        let sibling = bind_server_udp(bound, true).unwrap();
        assert_eq!(sibling.local_addr().unwrap(), bound);

        let other_primary = bind_server_udp(ephemeral, false).unwrap();
        assert_ne!(other_primary.local_addr().unwrap(), bound);
    }
}
