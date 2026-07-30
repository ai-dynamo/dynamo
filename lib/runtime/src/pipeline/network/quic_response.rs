// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-lane QUIC transport for streaming worker responses.
//!
//! A worker keeps a fixed connection bundle to a frontend and distributes a
//! fixed set of long-lived bidirectional streams (lanes) across it. Logical
//! response frames are request-hashed to a lane once, queued in a bounded
//! Tokio channel, and written in batches by the lane's sole writer task.

use std::{
    collections::HashMap,
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use bytes::{BufMut, Bytes, BytesMut};
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
use tokio::{
    io::AsyncReadExt,
    sync::{mpsc, oneshot},
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
const PROTOCOL_VERSION: u8 = 1;
const ALPN: &[u8] = b"dynamo-response-v1";
const DEFAULT_CONNECTIONS: usize = 8;
const MIN_CONNECTIONS: usize = 1;
const MAX_CONNECTIONS: usize = 8;
const DEFAULT_LANES: usize = 8;
const MIN_LANES: usize = 1;
const MAX_LANES: usize = 64;
const LANE_QUEUE_CAPACITY: usize = 512;
const MAX_BATCH_FRAMES: usize = 64;
const DEFAULT_BATCH_INTERVAL_US: u64 = 1_000;
const MAX_BATCH_INTERVAL_US: u64 = 100_000;
const FRAME_HEADER_LEN: usize = 1 + 16 + 4;
const MAX_FRAME_PAYLOAD: usize = 32 * 1024 * 1024;
const RESPONSE_BUFFER_CAPACITY: usize = 64;
const TOMBSTONE_TTL: Duration = Duration::from_secs(5);
const CLOSE_CODE_INVARIANT: quinn::VarInt = quinn::VarInt::from_u32(1);

#[derive(Debug, Clone, Copy)]
pub struct QuicResponseConfig {
    pub connections: usize,
    /// Total lanes across every connection in the bundle.
    pub lanes: usize,
    pub batch_interval: Duration,
}

impl QuicResponseConfig {
    pub fn from_env() -> Result<Self, PipelineError> {
        let connections = parse_env_range(
            quic_response::DYN_QUIC_RESPONSE_CONNECTIONS,
            DEFAULT_CONNECTIONS,
            MIN_CONNECTIONS,
            MAX_CONNECTIONS,
        )?;
        let lanes = parse_env_range(
            quic_response::DYN_QUIC_RESPONSE_LANES,
            DEFAULT_LANES,
            MIN_LANES,
            MAX_LANES,
        )?;
        let interval_us = parse_env_range(
            quic_response::DYN_QUIC_RESPONSE_BATCH_INTERVAL_US,
            DEFAULT_BATCH_INTERVAL_US,
            0,
            MAX_BATCH_INTERVAL_US,
        )?;
        if connections > lanes {
            return Err(PipelineError::Generic(format!(
                "{} ({connections}) cannot exceed {} ({lanes})",
                quic_response::DYN_QUIC_RESPONSE_CONNECTIONS,
                quic_response::DYN_QUIC_RESPONSE_LANES,
            )));
        }
        Ok(Self {
            connections,
            lanes,
            batch_interval: Duration::from_micros(interval_us),
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

async fn read_frame(recv: &mut quinn::RecvStream) -> Result<Frame> {
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
    Ok(Frame::new(kind, registration_id, payload.freeze()))
}

/// Fill `batch` with up to 64 frames. The first `recv_many` waits while idle;
/// subsequent cancellation-safe `recv_many` calls race the single batch
/// deadline. A zero interval therefore drains an already-ready burst but does
/// not intentionally wait for more work.
async fn receive_batch(
    receiver: &mut mpsc::Receiver<Frame>,
    batch: &mut Vec<Frame>,
    interval: Duration,
) -> Option<Duration> {
    batch.clear();
    if receiver.recv_many(batch, MAX_BATCH_FRAMES).await == 0 {
        return None;
    }

    let batch_started = Instant::now();
    if interval.is_zero() {
        return Some(batch_started.elapsed());
    }
    let deadline = Instant::now() + interval;
    while batch.len() < MAX_BATCH_FRAMES {
        tokio::select! {
            biased;
            _ = sleep_until(deadline) => break,
            count = receiver.recv_many(batch, MAX_BATCH_FRAMES - batch.len()) => {
                if count == 0 {
                    break;
                }
            }
        }
    }
    Some(batch_started.elapsed())
}

#[derive(Default)]
struct ServerState {
    pending: HashMap<Uuid, PendingResponse>,
    active: HashMap<Uuid, ActiveResponse>,
    registration_instance: HashMap<Uuid, EndpointInstanceId>,
    instance_registrations: HashMap<EndpointInstanceId, Vec<Uuid>>,
    removed_instances: HashMap<EndpointInstanceId, Instant>,
}

struct PendingResponse {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamReceiver, String>>,
}

struct ActiveResponse {
    sender: mpsc::Sender<Bytes>,
    monitor_cancel: CancellationToken,
    connection_id: usize,
}

fn prune_tombstones(state: &mut ServerState, now: Instant) {
    state
        .removed_instances
        .retain(|_, inserted| now.saturating_duration_since(*inserted) < TOMBSTONE_TTL);
}

pub struct QuicResponseServer {
    endpoint: quinn::Endpoint,
    advertised_address: SocketAddr,
    frontend_id: String,
    certificate_sha256: String,
    state: Arc<Mutex<ServerState>>,
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
        transport.max_concurrent_bidi_streams((MAX_LANES as u32).into());
        transport.max_concurrent_uni_streams(0_u8.into());
        transport.keep_alive_interval(Some(Duration::from_secs(5)));
        if let Some(window) = stream_receive_window {
            transport.stream_receive_window(quinn::VarInt::from_u32(window));
        }

        let endpoint = quinn::Endpoint::server(server_config, bind_address).map_err(|error| {
            PipelineError::Generic(format!(
                "failed binding QUIC response server on {bind_address}: {error}"
            ))
        })?;
        let advertised_address = if advertised_address.port() == 0 {
            endpoint.local_addr().map_err(|error| {
                PipelineError::Generic(format!("failed reading QUIC response address: {error}"))
            })?
        } else {
            advertised_address
        };
        let server = Arc::new(Self {
            endpoint,
            advertised_address,
            frontend_id: Uuid::new_v4().to_string(),
            certificate_sha256,
            state: Arc::new(Mutex::new(ServerState::default())),
            shutdown,
        });
        Self::spawn_accept_loop(&server);
        Ok(server)
    }

    fn spawn_accept_loop(server: &Arc<Self>) {
        let endpoint = server.endpoint.clone();
        let state = server.state.clone();
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
                                Ok(connection) => run_server_connection(connection, state).await,
                                Err(error) => tracing::warn!(%error, "QUIC response handshake failed"),
                            }
                        });
                    }
                }
            }
        });
    }

    pub fn register_response(
        &self,
        context: Arc<dyn AsyncEngineContext>,
    ) -> RegisteredStream<StreamReceiver> {
        let registration_id = Uuid::new_v4();
        let request_id = context.id().to_string();
        let (pending_tx, pending_rx) = oneshot::channel();
        self.state.lock().pending.insert(
            registration_id,
            PendingResponse {
                context,
                connection: pending_tx,
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
        RegisteredStream::new(connection_info, pending_rx).with_cleanup(move || {
            remove_registration(&state, registration_id);
        })
    }

    pub async fn associate_instance(&self, registration_id: Uuid, id: &EndpointInstanceId) -> bool {
        let mut state = self.state.lock();
        let now = Instant::now();
        prune_tombstones(&mut state, now);
        if state.removed_instances.contains_key(id) {
            drop(state);
            remove_registration(&self.state, registration_id);
            return false;
        }
        state
            .registration_instance
            .insert(registration_id, id.clone());
        state
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
            let mut state = self.state.lock();
            let now = Instant::now();
            prune_tombstones(&mut state, now);
            state.removed_instances.insert(id.clone(), now);
            state.instance_registrations.remove(id).unwrap_or_default()
        };
        let count = registrations.len();
        for registration_id in registrations {
            remove_registration(&self.state, registration_id);
        }
        count
    }

    pub async fn clear_instance_tombstone(&self, id: &EndpointInstanceId) {
        self.state.lock().removed_instances.remove(id);
    }
}

fn remove_registration(state: &Mutex<ServerState>, registration_id: Uuid) {
    let mut state = state.lock();
    state.pending.remove(&registration_id);
    if let Some(active) = state.active.remove(&registration_id) {
        active.monitor_cancel.cancel();
    }
    if let Some(instance) = state.registration_instance.remove(&registration_id)
        && let Some(registrations) = state.instance_registrations.get_mut(&instance)
    {
        registrations.retain(|candidate| *candidate != registration_id);
        if registrations.is_empty() {
            state.instance_registrations.remove(&instance);
        }
    }
}

async fn run_server_connection(connection: quinn::Connection, state: Arc<Mutex<ServerState>>) {
    let connection_id = connection.stable_id();
    crate::metrics::quic_response::track_connection(connection.clone());
    tracing::debug!(connection_id, remote = %connection.remote_address(), "QUIC response connection established");
    loop {
        match connection.accept_bi().await {
            Ok((send, recv)) => {
                let connection = connection.clone();
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(error) =
                        run_server_lane(send, recv, connection_id, state.clone()).await
                    {
                        tracing::warn!(connection_id, %error, "QUIC response lane failed; closing connection");
                        fail_server_connection(&state, connection_id);
                        connection.close(CLOSE_CODE_INVARIANT, b"response lane invariant failure");
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

fn fail_server_connection(state: &Mutex<ServerState>, connection_id: usize) {
    let doomed: Vec<Uuid> = state
        .lock()
        .active
        .iter()
        .filter_map(|(registration_id, active)| {
            (active.connection_id == connection_id).then_some(*registration_id)
        })
        .collect();
    for registration_id in doomed {
        remove_registration(state, registration_id);
    }
}

async fn run_server_lane(
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    connection_id: usize,
    state: Arc<Mutex<ServerState>>,
) -> Result<()> {
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
            process_server_frame(frame, connection_id, &state, &control_tx).await?;
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
    connection_id: usize,
    state: &Arc<Mutex<ServerState>>,
    control_tx: &mpsc::Sender<Frame>,
) -> Result<()> {
    match frame.kind {
        FrameKind::Prologue => {
            if !frame.payload.is_empty() {
                bail!("QUIC response Prologue frame carried a payload");
            }
            let Some(pending) = state.lock().pending.remove(&frame.registration_id) else {
                send_control(control_tx, FrameKind::Reset, frame.registration_id).await?;
                return Ok(());
            };
            let (response_tx, response_rx) = mpsc::channel(RESPONSE_BUFFER_CAPACITY);
            let monitor_cancel = CancellationToken::new();
            state.lock().active.insert(
                frame.registration_id,
                ActiveResponse {
                    sender: response_tx.clone(),
                    monitor_cancel: monitor_cancel.clone(),
                    connection_id,
                },
            );
            if pending
                .connection
                .send(Ok(StreamReceiver { rx: response_rx }))
                .is_err()
            {
                remove_registration(state, frame.registration_id);
                send_control(control_tx, FrameKind::Reset, frame.registration_id).await?;
                return Ok(());
            }

            spawn_response_monitor(
                pending.context,
                response_tx,
                frame.registration_id,
                control_tx.clone(),
                monitor_cancel,
            );
        }
        FrameKind::Error => {
            let error = String::from_utf8(frame.payload.to_vec())
                .context("QUIC response terminal error was not UTF-8")?;
            let pending = state.lock().pending.remove(&frame.registration_id);
            if let Some(pending) = pending {
                let _ = pending.connection.send(Err(error));
                remove_registration(state, frame.registration_id);
            } else {
                send_control(control_tx, FrameKind::Reset, frame.registration_id).await?;
            }
        }
        FrameKind::Data => {
            let sender = state
                .lock()
                .active
                .get(&frame.registration_id)
                .map(|active| active.sender.clone());
            match sender {
                Some(sender) if sender.send(frame.payload).await.is_ok() => {}
                _ => send_control(control_tx, FrameKind::Reset, frame.registration_id).await?,
            }
        }
        FrameKind::End => {
            if !frame.payload.is_empty() {
                bail!("QUIC response End frame carried a payload");
            }
            remove_registration(state, frame.registration_id);
        }
        FrameKind::Stop | FrameKind::Kill | FrameKind::Reset => {
            bail!("worker sent reverse-only control frame {:?}", frame.kind)
        }
    }
    Ok(())
}

fn spawn_response_monitor(
    context: Arc<dyn AsyncEngineContext>,
    response_tx: mpsc::Sender<Bytes>,
    registration_id: Uuid,
    control_tx: mpsc::Sender<Frame>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let killed = context.killed();
        let stopped = context.stopped();
        let closed = response_tx.closed();
        tokio::pin!(killed, stopped, closed);
        let mut stop_sent = false;
        loop {
            let kind = tokio::select! {
                _ = cancel.cancelled() => return,
                _ = &mut killed => FrameKind::Kill,
                _ = &mut stopped, if !stop_sent => FrameKind::Stop,
                _ = &mut closed => FrameKind::Reset,
            };
            if send_control(&control_tx, kind, registration_id)
                .await
                .is_err()
            {
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

async fn send_control(
    control_tx: &mpsc::Sender<Frame>,
    kind: FrameKind,
    registration_id: Uuid,
) -> Result<()> {
    control_tx
        .send(Frame::new(kind, registration_id, Bytes::new()))
        .await
        .map_err(|_| anyhow!("QUIC response reverse-control lane closed"))
}

struct Lane {
    index: usize,
    sender: mpsc::Sender<Frame>,
}

struct ClientConnectionBundle {
    connections: Arc<[quinn::Connection]>,
    lanes: Arc<[Arc<Lane>]>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    healthy: Arc<AtomicBool>,
}

impl ClientConnectionBundle {
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
            && self
                .connections
                .iter()
                .all(|connection| connection.close_reason().is_none())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConnectionKey {
    address: String,
    frontend_id: String,
    certificate_sha256: String,
}

pub struct QuicResponseClientPool {
    config: QuicResponseConfig,
    endpoints: Mutex<HashMap<bool, quinn::Endpoint>>,
    connections: tokio::sync::Mutex<HashMap<ConnectionKey, Arc<ClientConnectionBundle>>>,
}

static PROCESS_CLIENT_POOL: OnceLock<Arc<QuicResponseClientPool>> = OnceLock::new();

pub fn process_client_pool_from_env() -> Result<Arc<QuicResponseClientPool>, PipelineError> {
    if let Some(pool) = PROCESS_CLIENT_POOL.get() {
        return Ok(pool.clone());
    }
    let pool = QuicResponseClientPool::from_env()?;
    let _ = PROCESS_CLIENT_POOL.set(pool);
    Ok(PROCESS_CLIENT_POOL
        .get()
        .expect("QUIC response client pool was just initialized")
        .clone())
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
            endpoints: Mutex::new(HashMap::new()),
            connections: tokio::sync::Mutex::new(HashMap::new()),
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
        let lane = connection.lanes[lane_index].clone();
        connection.contexts.lock().insert(
            info.registration_id,
            Arc::new(ClientResponseContext {
                context,
                cancellation_counter,
                cancellation_recorded: AtomicBool::new(false),
            }),
        );
        Ok(QuicResponseSender {
            lane,
            contexts: connection.contexts.clone(),
            registration_id: info.registration_id,
            prologue_sent: false,
            terminated: false,
        })
    }

    async fn connection(&self, key: ConnectionKey) -> Result<Arc<ClientConnectionBundle>> {
        // Holding this mutex through connection creation deliberately serializes
        // replacement. The path is cold and guarantees exactly one published
        // connection bundle/lane set for a frontend identity.
        let mut connections = self.connections.lock().await;
        if let Some(connection) = connections.get(&key)
            && connection.is_healthy()
        {
            return Ok(connection.clone());
        }
        connections.remove(&key);
        let connection = self.connect(&key).await?;
        connections.insert(key, connection.clone());
        Ok(connection)
    }

    async fn connect(&self, key: &ConnectionKey) -> Result<Arc<ClientConnectionBundle>> {
        let address: SocketAddr = key
            .address
            .parse()
            .with_context(|| format!("invalid QUIC response address {}", key.address))?;
        let expected_fingerprint = decode_fingerprint(&key.certificate_sha256)?;
        let client_config = pinned_client_config(expected_fingerprint)?;
        let endpoint = self.client_endpoint(address.is_ipv6())?;
        let mut connections = Vec::with_capacity(self.config.connections);
        for _ in 0..self.config.connections {
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

        let contexts = Arc::new(Mutex::new(HashMap::new()));
        let healthy = Arc::new(AtomicBool::new(true));
        let mut lanes = Vec::with_capacity(self.config.lanes);
        for index in 0..self.config.lanes {
            let connection = connections[index % connections.len()].clone();
            let (send, recv) = connection.open_bi().await?;
            let (lane_tx, lane_rx) = mpsc::channel(LANE_QUEUE_CAPACITY);
            let lane = Arc::new(Lane {
                index,
                sender: lane_tx,
            });
            spawn_client_lane(
                send,
                recv,
                lane_rx,
                self.config.batch_interval,
                connections.clone(),
                contexts.clone(),
                healthy.clone(),
            );
            lanes.push(lane);
        }

        tracing::debug!(
            remote = %address,
            connections = self.config.connections,
            total_lanes = self.config.lanes,
            "QUIC response connection bundle and lanes ready"
        );
        Ok(Arc::new(ClientConnectionBundle {
            connections,
            lanes: lanes.into(),
            contexts,
            healthy,
        }))
    }

    fn client_endpoint(&self, ipv6: bool) -> Result<quinn::Endpoint> {
        let mut endpoints = self.endpoints.lock();
        if let Some(endpoint) = endpoints.get(&ipv6) {
            return Ok(endpoint.clone());
        }
        let bind = if ipv6 {
            SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0)
        } else {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
        };
        let endpoint = quinn::Endpoint::client(bind)?;
        endpoints.insert(ipv6, endpoint.clone());
        Ok(endpoint)
    }
}

fn spawn_client_lane(
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    receiver: mpsc::Receiver<Frame>,
    interval: Duration,
    connections: Arc<[quinn::Connection]>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    healthy: Arc<AtomicBool>,
) {
    let writer_connections = connections.clone();
    let writer_contexts = contexts.clone();
    let writer_healthy = healthy.clone();
    tokio::spawn(async move {
        if let Err(error) =
            run_client_writer(send, receiver, interval, writer_contexts.clone()).await
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
    mut receiver: mpsc::Receiver<Frame>,
    interval: Duration,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
) -> Result<()> {
    let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
    let mut chunks = Vec::with_capacity(MAX_BATCH_FRAMES * 2);
    while let Some(batch_wait) = receive_batch(&mut receiver, &mut batch, interval).await {
        chunks.clear();
        for frame in &batch {
            chunks.push(frame.header());
            if !frame.payload.is_empty() {
                chunks.push(frame.payload.clone());
            }
        }
        send.write_all_chunks(&mut chunks).await?;
        crate::metrics::quic_response::BATCHES.inc();
        crate::metrics::quic_response::FRAMES_PER_BATCH.observe(batch.len() as f64);
        crate::metrics::quic_response::BATCH_WAIT_SECONDS.observe(batch_wait.as_secs_f64());
        if batch
            .iter()
            .any(|frame| matches!(frame.kind, FrameKind::Error | FrameKind::End))
        {
            let mut contexts = contexts.lock();
            for frame in &batch {
                if matches!(frame.kind, FrameKind::Error | FrameKind::End) {
                    contexts.remove(&frame.registration_id);
                }
            }
        }
        batch.clear();
    }
    bail!("QUIC response lane queue closed unexpectedly")
}

async fn run_client_control_reader(
    mut recv: quinn::RecvStream,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
) -> Result<()> {
    loop {
        let frame = read_frame(&mut recv).await?;
        if !frame.payload.is_empty() {
            bail!("QUIC response reverse control carried a payload");
        }
        let entry = contexts.lock().get(&frame.registration_id).cloned();
        match frame.kind {
            FrameKind::Stop => {
                if let Some(entry) = entry {
                    entry.record_cancellation();
                    entry.context.stop();
                }
            }
            FrameKind::Kill | FrameKind::Reset => {
                if let Some(entry) = entry {
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
    tracing::warn!(%reason, "QUIC response connection bundle invariant failed");
    for (_, entry) in contexts.lock().drain() {
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
    // Lane selection is performed exactly once by the pool. Token writes only
    // touch this Arc and its bounded channel.
    lane: Arc<Lane>,
    contexts: Arc<Mutex<HashMap<Uuid, Arc<ClientResponseContext>>>>,
    registration_id: Uuid,
    prologue_sent: bool,
    terminated: bool,
}

struct ClientResponseContext {
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<IntCounter>,
    cancellation_recorded: AtomicBool,
}

impl ClientResponseContext {
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
        self.enqueue(Frame::new(kind, self.registration_id, payload))
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
        self.enqueue(Frame::new(FrameKind::Data, self.registration_id, payload))
            .await
    }

    pub async fn finish(&mut self) -> Result<()> {
        if !self.prologue_sent || self.terminated {
            return Ok(());
        }
        self.enqueue(Frame::new(
            FrameKind::End,
            self.registration_id,
            Bytes::new(),
        ))
        .await?;
        self.terminated = true;
        Ok(())
    }

    async fn enqueue(&self, frame: Frame) -> Result<()> {
        if frame.payload.len() > MAX_FRAME_PAYLOAD {
            bail!(
                "QUIC response frame payload {} exceeds {}",
                frame.payload.len(),
                MAX_FRAME_PAYLOAD
            );
        }
        match self.lane.sender.try_send(frame) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                bail!("QUIC response lane closed")
            }
            Err(mpsc::error::TrySendError::Full(frame)) => {
                let blocked_at = Instant::now();
                let result = self.lane.sender.send(frame).await;
                crate::metrics::quic_response::BLOCKED_ENQUEUE_SECONDS
                    .observe(blocked_at.elapsed().as_secs_f64());
                result.map_err(|_| anyhow!("QUIC response lane closed"))?;
                tracing::trace!(
                    blocked_seconds = blocked_at.elapsed().as_secs_f64(),
                    "QUIC response enqueue blocked"
                );
                Ok(())
            }
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
        let registration_id = self.registration_id;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
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
    if encoded.len() != 64 {
        bail!("invalid SHA-256 fingerprint length {}", encoded.len());
    }
    let mut decoded = [0_u8; 32];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16)
            .context("invalid SHA-256 fingerprint")?;
    }
    Ok(decoded)
}

pub fn registration_id(connection_info: &ConnectionInfo) -> Option<Uuid> {
    if connection_info.transport != TRANSPORT_NAME {
        return None;
    }
    serde_json::from_str::<QuicResponseConnectionInfo>(&connection_info.info)
        .ok()
        .map(|info| info.registration_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::AsyncEngineContextProvider, pipeline::Context as PipelineContext};

    fn frame(index: usize) -> Frame {
        Frame::new(FrameKind::Data, Uuid::nil(), Bytes::from(index.to_string()))
    }

    fn test_pool(connections: usize, lanes: usize) -> QuicResponseClientPool {
        QuicResponseClientPool {
            config: QuicResponseConfig {
                connections,
                lanes,
                batch_interval: Duration::ZERO,
            },
            endpoints: Mutex::new(HashMap::new()),
            connections: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    fn context_for_lane(lane: usize, lanes: usize) -> PipelineContext<()> {
        for candidate in 0_u64.. {
            let id = format!("lane-{lane}-{candidate}");
            if xxh3_64(id.as_bytes()) as usize % lanes == lane {
                return PipelineContext::with_id_and_metadata((), id, Default::default());
            }
        }
        unreachable!()
    }

    #[tokio::test]
    async fn bulk_drains_available_frames_at_key_boundaries() {
        for available in [1, 63, 64] {
            let (tx, mut rx) = mpsc::channel(128);
            for index in 0..available {
                tx.send(frame(index)).await.unwrap();
            }
            let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
            assert!(
                receive_batch(&mut rx, &mut batch, Duration::ZERO)
                    .await
                    .is_some()
            );
            assert_eq!(batch.len(), available);
        }
    }

    #[tokio::test]
    async fn deadline_flushes_partial_batch_and_reuses_vector() {
        let (tx, mut rx) = mpsc::channel(128);
        tx.send(frame(0)).await.unwrap();
        let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
        let allocation = batch.as_ptr();
        assert!(
            receive_batch(&mut rx, &mut batch, Duration::from_millis(1))
                .await
                .is_some()
        );
        assert_eq!(batch.len(), 1);
        batch.clear();
        assert_eq!(batch.as_ptr(), allocation);
    }

    #[tokio::test(start_paused = true)]
    async fn frames_arriving_before_deadline_join_the_same_batch() {
        let (tx, mut rx) = mpsc::channel(128);
        tx.send(frame(0)).await.unwrap();
        let task = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
            let wait = receive_batch(&mut rx, &mut batch, Duration::from_millis(1)).await;
            (wait, batch.len())
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_micros(500)).await;
        for index in 1..64 {
            tx.send(frame(index)).await.unwrap();
        }
        tokio::task::yield_now().await;
        let (wait, len) = task.await.unwrap();
        assert!(wait.is_some());
        assert_eq!(len, 64);
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_wins_when_a_frame_arrives_at_the_boundary() {
        let (tx, mut rx) = mpsc::channel(128);
        tx.send(frame(0)).await.unwrap();
        let task = tokio::spawn(async move {
            let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
            let wait = receive_batch(&mut rx, &mut batch, Duration::from_millis(1)).await;
            (wait, batch, rx)
        });
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_millis(1)).await;
        tx.send(frame(1)).await.unwrap();
        tokio::task::yield_now().await;

        let (wait, batch, mut rx) = task.await.unwrap();
        assert!(wait.is_some());
        assert_eq!(batch.len(), 1);
        assert_eq!(rx.recv().await.unwrap().payload, Bytes::from_static(b"1"));
    }

    #[tokio::test]
    async fn closed_channel_returns_final_batch_then_eof() {
        let (tx, mut rx) = mpsc::channel(8);
        tx.send(frame(0)).await.unwrap();
        drop(tx);
        let mut batch = Vec::with_capacity(MAX_BATCH_FRAMES);
        assert!(
            receive_batch(&mut rx, &mut batch, Duration::ZERO)
                .await
                .is_some()
        );
        assert_eq!(batch.len(), 1);
        assert!(
            receive_batch(&mut rx, &mut batch, Duration::ZERO)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn bounded_lane_queue_blocks_only_after_512_frames() {
        let (tx, mut rx) = mpsc::channel(LANE_QUEUE_CAPACITY);
        for index in 0..LANE_QUEUE_CAPACITY {
            tx.try_send(frame(index)).unwrap();
        }
        let blocked = tokio::spawn({
            let tx = tx.clone();
            async move { tx.send(frame(LANE_QUEUE_CAPACITY)).await }
        });
        tokio::task::yield_now().await;
        assert!(!blocked.is_finished());
        for expected in 0..=LANE_QUEUE_CAPACITY {
            let received = rx.recv().await.unwrap();
            assert_eq!(received.payload, Bytes::from(expected.to_string()));
        }
        blocked.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn eight_connections_eight_lanes_carry_1000_ordered_responses() {
        let shutdown = CancellationToken::new();
        let server = QuicResponseServer::new(
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            shutdown.clone(),
        )
        .unwrap();
        let pool = test_pool(8, 8);

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
        let connections = pool.connections.lock().await;
        assert_eq!(connections.len(), 1);
        let bundle = connections.values().next().unwrap();
        assert_eq!(bundle.connections.len(), 8);
        assert_eq!(bundle.lanes.len(), 8);
        shutdown.cancel();
    }

    #[tokio::test]
    async fn receiver_drop_sends_logical_reset_without_resetting_lane() {
        let shutdown = CancellationToken::new();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = QuicResponseServer::new(address, address, shutdown.clone()).unwrap();
        let pool = test_pool(1, 4);
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
        let connections = pool.connections.lock().await;
        assert!(connections.values().next().unwrap().is_healthy());
        drop(connections);

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
    async fn terminal_generate_error_uses_ordered_lane_frame() {
        let shutdown = CancellationToken::new();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = QuicResponseServer::new(address, address, shutdown.clone()).unwrap();
        let pool = test_pool(1, 4);
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
    async fn physical_connection_failure_kills_all_and_reconnects_once() {
        let shutdown = CancellationToken::new();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = QuicResponseServer::new(address, address, shutdown.clone()).unwrap();
        let pool = Arc::new(test_pool(2, 8));

        let first_context = PipelineContext::new(());
        let first = server.register_response(first_context.context());
        let first_info = first.connection_info.clone();
        let mut first_sender = pool
            .sender(first_context.context(), first_info.clone())
            .await
            .unwrap();
        first_sender.send_prologue(None).await.unwrap();
        let old_connections = {
            let connections = pool.connections.lock().await;
            connections
                .values()
                .next()
                .unwrap()
                .connections
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let old_ids = old_connections
            .iter()
            .map(quinn::Connection::stable_id)
            .collect::<Vec<_>>();
        old_connections[0].close(CLOSE_CODE_INVARIANT, b"test lane failure");
        let first_engine_context = first_context.context();
        tokio::time::timeout(Duration::from_secs(1), first_engine_context.killed())
            .await
            .expect("connection failure should kill every active context");

        let mut registrations = Vec::new();
        for _ in 0..16 {
            let context = PipelineContext::new(());
            let registered = server.register_response(context.context());
            registrations.push((context, registered.connection_info));
        }
        let senders =
            futures::future::join_all(registrations.into_iter().map(|(context, info)| {
                let pool = pool.clone();
                async move { pool.sender(context.context(), info).await }
            }))
            .await;
        assert!(senders.iter().all(Result::is_ok));
        let connections = pool.connections.lock().await;
        assert_eq!(connections.len(), 1);
        let replacement = connections.values().next().unwrap();
        assert_eq!(replacement.connections.len(), 2);
        assert!(
            replacement
                .connections
                .iter()
                .all(|connection| !old_ids.contains(&connection.stable_id()))
        );
        assert_eq!(replacement.lanes.len(), 8);
        shutdown.cancel();
    }

    #[tokio::test]
    async fn stalled_lane_does_not_block_sibling_lane() {
        let shutdown = CancellationToken::new();
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let server = QuicResponseServer::new_with_stream_window(
            address,
            address,
            shutdown.clone(),
            Some(8 * 1024),
        )
        .unwrap();
        let pool = Arc::new(test_pool(1, 4));

        let stalled_context = context_for_lane(0, 4);
        let stalled = server.register_response(stalled_context.context());
        let (stalled_info, stalled_provider) = stalled.into_parts();
        let mut stalled_sender = pool
            .sender(stalled_context.context(), stalled_info)
            .await
            .unwrap();
        assert_eq!(stalled_sender.lane_index(), 0);
        stalled_sender.send_prologue(None).await.unwrap();
        let _stalled_receiver = stalled_provider.await.unwrap().unwrap();
        let stall_task = tokio::spawn(async move {
            let payload = Bytes::from(vec![0x5a; 16 * 1024]);
            for _ in 0..2_000 {
                stalled_sender.send(payload.clone()).await?;
            }
            Ok::<_, anyhow::Error>(())
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !stall_task.is_finished(),
            "stalled lane producer should back up"
        );

        let sibling_context = context_for_lane(1, 4);
        let sibling = server.register_response(sibling_context.context());
        let (sibling_info, sibling_provider) = sibling.into_parts();
        let mut sibling_sender = pool
            .sender(sibling_context.context(), sibling_info)
            .await
            .unwrap();
        assert_eq!(sibling_sender.lane_index(), 1);
        sibling_sender.send_prologue(None).await.unwrap();
        sibling_sender
            .send(Bytes::from_static(b"progress"))
            .await
            .unwrap();
        sibling_sender.finish().await.unwrap();
        let mut sibling_receiver = tokio::time::timeout(Duration::from_secs(1), sibling_provider)
            .await
            .expect("sibling prologue should progress")
            .unwrap()
            .unwrap();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), sibling_receiver.rx.recv())
                .await
                .expect("sibling data should progress")
                .unwrap(),
            Bytes::from_static(b"progress")
        );
        stall_task.abort();
        shutdown.cancel();
    }

    #[test]
    fn fingerprint_round_trip() {
        let fingerprint = [0x5a; 32];
        assert_eq!(
            decode_fingerprint(&encode_hex(&fingerprint)).unwrap(),
            fingerprint
        );
    }
}
