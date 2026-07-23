// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response-only QUIC transport.
//!
//! A frontend owns one QUIC endpoint and advertises a pinned ephemeral
//! certificate. Workers reuse one connection to that endpoint and open one
//! bidirectional QUIC stream per logical response. The worker-to-frontend half
//! carries the response prologue and length-prefixed payloads; the reverse half
//! carries cancellation controls.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::{IpAddr, SocketAddr},
    sync::{
        Arc, Mutex as StdMutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result, bail};
use bytes::{Bytes, BytesMut};
use dashmap::DashMap;
use parking_lot::Mutex;
use quinn::{Connection, Endpoint, RecvStream, SendStream, TransportConfig, VarInt};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::{io::AsyncReadExt, sync::oneshot, time};
use uuid::Uuid;

use super::{ConnectionInfo, RegisteredStream, StreamOptions, StreamReceiver};
use crate::{discovery::EndpointInstanceId, engine::AsyncEngineContext};

pub const ALPN: &[u8] = b"dynamo-response-v1";
pub const PROTOCOL_VERSION: u16 = 1;

const STREAM_RECEIVE_WINDOW: u32 = 256 * 1024;
const CONNECTION_WINDOW: u32 = 8 * 1024 * 1024;
const SEND_WINDOW: u64 = 8 * 1024 * 1024;
const HIGH_STREAM_RECEIVE_WINDOW: u32 = 1024 * 1024;
const HIGH_CONNECTION_WINDOW: u32 = 32 * 1024 * 1024;
const HIGH_SEND_WINDOW: u64 = 32 * 1024 * 1024;
const MAX_BIDI_STREAMS: u32 = 4096;
const UDP_SOCKET_BUFFER: usize = 8 * 1024 * 1024;
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_ERROR_BYTES: usize = 64 * 1024;
const TOMBSTONE_TTL: Duration = Duration::from_secs(5);

const STATUS_SUCCESS: u8 = 0;
const STATUS_ERROR: u8 = 1;
const CONTROL_STOP: u8 = 1;
const CONTROL_KILL: u8 = 2;

// Stable application error codes. These are intentionally transport-local and
// must not be generated dynamically from error strings.
const RESET_MALFORMED: u32 = 0x100;
const RESET_UNKNOWN_REGISTRATION: u32 = 0x101;
const RESET_OVERSIZED_FRAME: u32 = 0x103;
const RESET_RECEIVER_DROPPED: u32 = 0x104;
const RESET_PUBLISHER_DROPPED: u32 = 0x105;
const RESET_KILLED: u32 = 0x106;

fn reset_code(code: u32) -> VarInt {
    VarInt::from_u32(code)
}

fn validate_payload_size(length: usize) -> Result<()> {
    let max = super::get_tcp_max_message_size().min(u32::MAX as usize);
    if length > max {
        bail!("QUIC response frame of {length} bytes exceeds {max}");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuicResponseConnectionInfo {
    pub protocol_version: u16,
    pub address: String,
    pub frontend_server_id: Uuid,
    pub response_registration_id: Uuid,
    pub context: String,
    pub certificate_fingerprint: [u8; 32],
}

impl QuicResponseConnectionInfo {
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != PROTOCOL_VERSION {
            bail!(
                "unsupported QUIC response protocol version {}; expected {}",
                self.protocol_version,
                PROTOCOL_VERSION
            );
        }
        self.address
            .parse::<SocketAddr>()
            .context("invalid QUIC response address")?;
        if self.frontend_server_id.is_nil() {
            bail!("QUIC frontend server ID must not be nil");
        }
        if self.response_registration_id.is_nil() {
            bail!("QUIC response registration ID must not be nil");
        }
        if self.context.is_empty() {
            bail!("QUIC response context must not be empty");
        }
        if self.certificate_fingerprint == [0; 32] {
            bail!("QUIC certificate fingerprint must not be all zeroes");
        }
        Ok(())
    }
}

impl From<QuicResponseConnectionInfo> for ConnectionInfo {
    fn from(info: QuicResponseConnectionInfo) -> Self {
        Self {
            transport: "quic_response".to_string(),
            info: serde_json::to_string(&info).expect("QuicResponseConnectionInfo must serialize"),
        }
    }
}

impl TryFrom<ConnectionInfo> for QuicResponseConnectionInfo {
    type Error = anyhow::Error;

    fn try_from(info: ConnectionInfo) -> Result<Self> {
        if info.transport != "quic_response" {
            bail!(
                "invalid transport {}; expected quic_response",
                info.transport
            );
        }
        let info: Self = serde_json::from_str(&info.info)
            .context("failed to decode QUIC response connection info")?;
        info.validate()?;
        Ok(info)
    }
}

/// Process-wide counters whose values can be sampled around a locked benchmark
/// window. Quinn connection statistics are exposed separately by
/// [`client_connection_stats`].
#[derive(Debug, Default, Clone, Serialize)]
pub struct QuicResponseCounters {
    pub handshakes: u64,
    pub handshake_failures: u64,
    pub reconnects: u64,
    pub connection_failures: u64,
    pub stream_open_stalls: u64,
    pub active_connections: u64,
    pub active_streams: u64,
    pub response_payloads: u64,
    pub malformed_resets: u64,
    pub unknown_registration_resets: u64,
    pub receiver_dropped_resets: u64,
    pub publisher_dropped_resets: u64,
    pub oversized_frame_resets: u64,
    pub killed_resets: u64,
}

#[derive(Default)]
struct Counters {
    handshakes: AtomicU64,
    handshake_failures: AtomicU64,
    reconnects: AtomicU64,
    connection_failures: AtomicU64,
    stream_open_stalls: AtomicU64,
    active_connections: AtomicU64,
    active_streams: AtomicU64,
    response_payloads: AtomicU64,
    malformed_resets: AtomicU64,
    unknown_registration_resets: AtomicU64,
    receiver_dropped_resets: AtomicU64,
    publisher_dropped_resets: AtomicU64,
    oversized_frame_resets: AtomicU64,
    killed_resets: AtomicU64,
}

static COUNTERS: Counters = Counters {
    handshakes: AtomicU64::new(0),
    handshake_failures: AtomicU64::new(0),
    reconnects: AtomicU64::new(0),
    connection_failures: AtomicU64::new(0),
    stream_open_stalls: AtomicU64::new(0),
    active_connections: AtomicU64::new(0),
    active_streams: AtomicU64::new(0),
    response_payloads: AtomicU64::new(0),
    malformed_resets: AtomicU64::new(0),
    unknown_registration_resets: AtomicU64::new(0),
    receiver_dropped_resets: AtomicU64::new(0),
    publisher_dropped_resets: AtomicU64::new(0),
    oversized_frame_resets: AtomicU64::new(0),
    killed_resets: AtomicU64::new(0),
};
static CLIENT_UDP_SEND_BUFFER: AtomicU64 = AtomicU64::new(0);
static CLIENT_UDP_RECV_BUFFER: AtomicU64 = AtomicU64::new(0);
static SERVER_UDP_SEND_BUFFER: AtomicU64 = AtomicU64::new(0);
static SERVER_UDP_RECV_BUFFER: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct ClosedConnectionCounters {
    udp_tx_bytes: AtomicU64,
    udp_tx_datagrams: AtomicU64,
    udp_tx_ios: AtomicU64,
    udp_rx_bytes: AtomicU64,
    udp_rx_datagrams: AtomicU64,
    udp_rx_ios: AtomicU64,
    stream_frames_tx: AtomicU64,
    connection_blocked_frames_tx: AtomicU64,
    stream_blocked_frames_tx: AtomicU64,
    lost_packets: AtomicU64,
    lost_bytes: AtomicU64,
}

static CLOSED_CONNECTION_COUNTERS: ClosedConnectionCounters = ClosedConnectionCounters {
    udp_tx_bytes: AtomicU64::new(0),
    udp_tx_datagrams: AtomicU64::new(0),
    udp_tx_ios: AtomicU64::new(0),
    udp_rx_bytes: AtomicU64::new(0),
    udp_rx_datagrams: AtomicU64::new(0),
    udp_rx_ios: AtomicU64::new(0),
    stream_frames_tx: AtomicU64::new(0),
    connection_blocked_frames_tx: AtomicU64::new(0),
    stream_blocked_frames_tx: AtomicU64::new(0),
    lost_packets: AtomicU64::new(0),
    lost_bytes: AtomicU64::new(0),
};

fn record_closed_connection(connection: &Connection) {
    let stats = connection.stats();
    macro_rules! add {
        ($field:ident, $value:expr) => {
            CLOSED_CONNECTION_COUNTERS
                .$field
                .fetch_add($value, Ordering::Relaxed)
        };
    }
    add!(udp_tx_bytes, stats.udp_tx.bytes);
    add!(udp_tx_datagrams, stats.udp_tx.datagrams);
    add!(udp_tx_ios, stats.udp_tx.ios);
    add!(udp_rx_bytes, stats.udp_rx.bytes);
    add!(udp_rx_datagrams, stats.udp_rx.datagrams);
    add!(udp_rx_ios, stats.udp_rx.ios);
    add!(stream_frames_tx, stats.frame_tx.stream);
    add!(connection_blocked_frames_tx, stats.frame_tx.data_blocked);
    add!(stream_blocked_frames_tx, stats.frame_tx.stream_data_blocked);
    add!(lost_packets, stats.path.lost_packets);
    add!(lost_bytes, stats.path.lost_bytes);
}

pub fn counters() -> QuicResponseCounters {
    QuicResponseCounters {
        handshakes: COUNTERS.handshakes.load(Ordering::Relaxed),
        handshake_failures: COUNTERS.handshake_failures.load(Ordering::Relaxed),
        reconnects: COUNTERS.reconnects.load(Ordering::Relaxed),
        connection_failures: COUNTERS.connection_failures.load(Ordering::Relaxed),
        stream_open_stalls: COUNTERS.stream_open_stalls.load(Ordering::Relaxed),
        active_connections: COUNTERS.active_connections.load(Ordering::Relaxed),
        active_streams: COUNTERS.active_streams.load(Ordering::Relaxed),
        response_payloads: COUNTERS.response_payloads.load(Ordering::Relaxed),
        malformed_resets: COUNTERS.malformed_resets.load(Ordering::Relaxed),
        unknown_registration_resets: COUNTERS.unknown_registration_resets.load(Ordering::Relaxed),
        receiver_dropped_resets: COUNTERS.receiver_dropped_resets.load(Ordering::Relaxed),
        publisher_dropped_resets: COUNTERS.publisher_dropped_resets.load(Ordering::Relaxed),
        oversized_frame_resets: COUNTERS.oversized_frame_resets.load(Ordering::Relaxed),
        killed_resets: COUNTERS.killed_resets.load(Ordering::Relaxed),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuicConnectionStats {
    pub address: String,
    pub frontend_server_id: Uuid,
    pub udp_tx_bytes: u64,
    pub udp_tx_datagrams: u64,
    pub udp_tx_ios: u64,
    pub udp_rx_bytes: u64,
    pub udp_rx_datagrams: u64,
    pub udp_rx_ios: u64,
    pub stream_frames_tx: u64,
    pub connection_blocked_frames_tx: u64,
    pub stream_blocked_frames_tx: u64,
    pub rtt_micros: u64,
    pub mtu: u16,
    pub lost_packets: u64,
    pub lost_bytes: u64,
}

/// Direct response sender: there is no application mpsc between the engine and
/// Quinn. Each `send` makes the frame available to Quinn immediately.
pub struct ResponseStreamSender {
    send: SendStream,
    payload_count: u64,
    prologue_sent: bool,
    payloads_allowed: bool,
    finished: bool,
}

impl ResponseStreamSender {
    async fn new(mut send: SendStream, registration_id: Uuid) -> Result<Self> {
        send.write_all(&registration_id.as_u128().to_be_bytes())
            .await
            .context("failed to write QUIC response registration ID")?;
        Ok(Self {
            send,
            payload_count: 0,
            prologue_sent: false,
            payloads_allowed: false,
            finished: false,
        })
    }

    pub async fn send_prologue(&mut self, error: Option<String>) -> Result<(), String> {
        if self.prologue_sent {
            return Err("QUIC response prologue already sent".to_string());
        }
        let (status, error) = match error {
            Some(error) => (STATUS_ERROR, error.into_bytes()),
            None => (STATUS_SUCCESS, Vec::new()),
        };
        if error.len() > MAX_ERROR_BYTES || error.len() > u32::MAX as usize {
            return Err(format!(
                "QUIC response prologue error exceeds {MAX_ERROR_BYTES} bytes"
            ));
        }
        let mut header = [0u8; 5];
        header[0] = status;
        header[1..].copy_from_slice(&(error.len() as u32).to_be_bytes());
        let mut chunks = [Bytes::copy_from_slice(&header), Bytes::from(error)];
        self.send
            .write_all_chunks(&mut chunks)
            .await
            .map_err(|e| e.to_string())?;
        self.prologue_sent = true;
        self.payloads_allowed = status == STATUS_SUCCESS;
        Ok(())
    }

    pub async fn send(&mut self, data: Bytes) -> Result<()> {
        if !self.prologue_sent {
            bail!("QUIC response payload sent before prologue");
        }
        if !self.payloads_allowed {
            bail!("QUIC error response cannot carry payload frames");
        }
        if let Err(error) = validate_payload_size(data.len()) {
            self.reset(RESET_OVERSIZED_FRAME);
            return Err(error);
        }
        let length = Bytes::copy_from_slice(&(data.len() as u32).to_be_bytes());
        let mut chunks = [length, data];
        self.send
            .write_all_chunks(&mut chunks)
            .await
            .context("failed to write QUIC response frame")?;
        self.payload_count += 1;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.prologue_sent {
            bail!("QUIC response finished before prologue");
        }
        if !self.finished {
            self.send
                .finish()
                .context("failed to finish QUIC response")?;
            self.finished = true;
        }
        Ok(())
    }

    fn reset(&mut self, code: u32) {
        let _ = self.send.reset(reset_code(code));
        if code == RESET_OVERSIZED_FRAME {
            COUNTERS
                .oversized_frame_resets
                .fetch_add(1, Ordering::Relaxed);
        }
        self.finished = true;
    }
}

impl Drop for ResponseStreamSender {
    fn drop(&mut self) {
        COUNTERS
            .response_payloads
            .fetch_add(self.payload_count, Ordering::Relaxed);
        COUNTERS.active_streams.fetch_sub(1, Ordering::Relaxed);
        if !self.finished {
            let _ = self.send.reset(reset_code(RESET_PUBLISHER_DROPPED));
            COUNTERS
                .publisher_dropped_resets
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConnectionKey {
    address: SocketAddr,
    frontend_server_id: Uuid,
    certificate_fingerprint: [u8; 32],
    protocol_version: u16,
}

struct ClientPool {
    endpoints: StdMutex<HashMap<bool, Endpoint>>,
    connections: Arc<DashMap<ConnectionKey, Connection>>,
    connecting: DashMap<ConnectionKey, Arc<tokio::sync::Mutex<()>>>,
    qlog: Option<quinn::QlogStream>,
}

impl ClientPool {
    fn new() -> Self {
        Self::new_with_qlog(None)
    }

    fn new_with_qlog(qlog: Option<quinn::QlogStream>) -> Self {
        Self {
            endpoints: StdMutex::new(HashMap::new()),
            connections: Arc::new(DashMap::new()),
            connecting: DashMap::new(),
            qlog,
        }
    }

    fn endpoint(&self, ipv6: bool) -> Result<Endpoint> {
        let mut endpoints = self.endpoints.lock().expect("QUIC endpoint mutex poisoned");
        if let Some(endpoint) = endpoints.get(&ipv6) {
            return Ok(endpoint.clone());
        }
        let bind = if ipv6 {
            "[::]:0".parse().expect("valid IPv6 wildcard")
        } else {
            "0.0.0.0:0".parse().expect("valid IPv4 wildcard")
        };
        let (socket, send_buffer, recv_buffer) = udp_socket(bind)?;
        CLIENT_UDP_SEND_BUFFER.store(send_buffer as u64, Ordering::Relaxed);
        CLIENT_UDP_RECV_BUFFER.store(recv_buffer as u64, Ordering::Relaxed);
        tracing::info!(
            requested_send_buffer = UDP_SOCKET_BUFFER,
            actual_send_buffer = send_buffer,
            requested_recv_buffer = UDP_SOCKET_BUFFER,
            actual_recv_buffer = recv_buffer,
            "configured QUIC client UDP socket buffers"
        );
        let endpoint = Endpoint::new(
            quinn::EndpointConfig::default(),
            None,
            socket,
            Arc::new(quinn::TokioRuntime),
        )?;
        endpoints.insert(ipv6, endpoint.clone());
        Ok(endpoint)
    }

    async fn connection(&self, info: &QuicResponseConnectionInfo) -> Result<Connection> {
        let address = info.address.parse::<SocketAddr>()?;
        let key = ConnectionKey {
            address,
            frontend_server_id: info.frontend_server_id,
            certificate_fingerprint: info.certificate_fingerprint,
            protocol_version: info.protocol_version,
        };
        if let Some(connection) = self.connections.get(&key)
            && connection.close_reason().is_none()
        {
            return Ok(connection.clone());
        }

        let connect_lock = self
            .connecting
            .entry(key.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let _guard = connect_lock.lock().await;
        if let Some(connection) = self.connections.get(&key)
            && connection.close_reason().is_none()
        {
            return Ok(connection.clone());
        }

        if let Some((_, closed)) = self.connections.remove(&key) {
            record_closed_connection(&closed);
            COUNTERS.reconnects.fetch_add(1, Ordering::Relaxed);
            COUNTERS.active_connections.fetch_sub(1, Ordering::Relaxed);
        }

        let endpoint = self.endpoint(address.is_ipv6())?;
        let config = client_config(info.certificate_fingerprint, self.qlog.clone())?;
        let connecting = endpoint
            .connect_with(config, address, "dynamo-response")
            .context("failed to start QUIC connection")?;
        let connection = match time::timeout(OPEN_TIMEOUT, connecting).await {
            Ok(Ok(connection)) => connection,
            Ok(Err(error)) => {
                COUNTERS.handshake_failures.fetch_add(1, Ordering::Relaxed);
                return Err(error.into());
            }
            Err(_) => {
                COUNTERS.handshake_failures.fetch_add(1, Ordering::Relaxed);
                bail!("timed out establishing QUIC response connection");
            }
        };
        COUNTERS.handshakes.fetch_add(1, Ordering::Relaxed);
        COUNTERS.active_connections.fetch_add(1, Ordering::Relaxed);
        self.connections.insert(key.clone(), connection.clone());
        let connections = self.connections.clone();
        let observed = connection.clone();
        tokio::spawn(async move {
            let _ = observed.closed().await;
            COUNTERS.connection_failures.fetch_add(1, Ordering::Relaxed);
            let should_remove = connections
                .get(&key)
                .is_some_and(|current| current.stable_id() == observed.stable_id());
            if should_remove && let Some((_, closed)) = connections.remove(&key) {
                record_closed_connection(&closed);
                COUNTERS.active_connections.fetch_sub(1, Ordering::Relaxed);
            }
        });
        Ok(connection)
    }

    fn invalidate(&self, info: &QuicResponseConnectionInfo) {
        let Ok(address) = info.address.parse::<SocketAddr>() else {
            return;
        };
        let key = ConnectionKey {
            address,
            frontend_server_id: info.frontend_server_id,
            certificate_fingerprint: info.certificate_fingerprint,
            protocol_version: info.protocol_version,
        };
        if let Some((_, closed)) = self.connections.remove(&key) {
            record_closed_connection(&closed);
            COUNTERS.active_connections.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn stats(&self) -> Vec<QuicConnectionStats> {
        self.connections
            .iter()
            .map(|entry| {
                let stats = entry.value().stats();
                QuicConnectionStats {
                    address: entry.key().address.to_string(),
                    frontend_server_id: entry.key().frontend_server_id,
                    udp_tx_bytes: stats.udp_tx.bytes,
                    udp_tx_datagrams: stats.udp_tx.datagrams,
                    udp_tx_ios: stats.udp_tx.ios,
                    udp_rx_bytes: stats.udp_rx.bytes,
                    udp_rx_datagrams: stats.udp_rx.datagrams,
                    udp_rx_ios: stats.udp_rx.ios,
                    stream_frames_tx: stats.frame_tx.stream,
                    connection_blocked_frames_tx: stats.frame_tx.data_blocked,
                    stream_blocked_frames_tx: stats.frame_tx.stream_data_blocked,
                    rtt_micros: stats.path.rtt.as_micros() as u64,
                    mtu: stats.path.current_mtu,
                    lost_packets: stats.path.lost_packets,
                    lost_bytes: stats.path.lost_bytes,
                }
            })
            .collect()
    }
}

static CLIENT_POOL: OnceLock<ClientPool> = OnceLock::new();
static QLOG_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct ProcessIdentity {
    frontend_server_id: Uuid,
    certificate: Vec<u8>,
    private_key: Vec<u8>,
    certificate_fingerprint: [u8; 32],
}

static PROCESS_IDENTITY: OnceLock<ProcessIdentity> = OnceLock::new();

fn process_identity() -> Result<&'static ProcessIdentity> {
    if let Some(identity) = PROCESS_IDENTITY.get() {
        return Ok(identity);
    }
    let generated = rcgen::generate_simple_self_signed(vec!["dynamo-response".to_string()])?;
    let certificate = generated.cert.der().to_vec();
    let identity = ProcessIdentity {
        frontend_server_id: Uuid::new_v4(),
        certificate_fingerprint: Sha256::digest(&certificate).into(),
        certificate,
        private_key: generated.key_pair.serialize_der(),
    };
    let _ = PROCESS_IDENTITY.set(identity);
    Ok(PROCESS_IDENTITY
        .get()
        .expect("QUIC process identity was just initialized"))
}

fn client_pool() -> &'static ClientPool {
    CLIENT_POOL.get_or_init(ClientPool::new)
}

pub fn client_connection_stats() -> Vec<QuicConnectionStats> {
    client_pool().stats()
}

/// Prometheus exposition generated at scrape time. All series are process
/// aggregates with either no labels or a fixed reset-reason label set.
pub fn prometheus_exposition() -> String {
    let counters = counters();
    let connections = client_connection_stats();
    let sum = |f: fn(&QuicConnectionStats) -> u64| connections.iter().map(f).sum::<u64>();
    let closed = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    let udp_tx_datagrams =
        sum(|stats| stats.udp_tx_datagrams) + closed(&CLOSED_CONNECTION_COUNTERS.udp_tx_datagrams);
    let udp_tx_ios = sum(|stats| stats.udp_tx_ios) + closed(&CLOSED_CONNECTION_COUNTERS.udp_tx_ios);
    let stream_frames =
        sum(|stats| stats.stream_frames_tx) + closed(&CLOSED_CONNECTION_COUNTERS.stream_frames_tx);
    let packet_density = if udp_tx_datagrams == 0 {
        0.0
    } else {
        counters.response_payloads as f64 / udp_tx_datagrams as f64
    };
    let stream_frame_density = if udp_tx_datagrams == 0 {
        0.0
    } else {
        stream_frames as f64 / udp_tx_datagrams as f64
    };
    let datagrams_per_io = if udp_tx_ios == 0 {
        0.0
    } else {
        udp_tx_datagrams as f64 / udp_tx_ios as f64
    };
    let rtt_micros = connections
        .iter()
        .map(|stats| stats.rtt_micros)
        .max()
        .unwrap_or_default();
    let mtu = connections
        .iter()
        .map(|stats| stats.mtu as u64)
        .min()
        .unwrap_or_default();

    format!(
        concat!(
            "# TYPE dynamo_quic_response_active_connections gauge\n",
            "dynamo_quic_response_active_connections {}\n",
            "# TYPE dynamo_quic_response_active_streams gauge\n",
            "dynamo_quic_response_active_streams {}\n",
            "# TYPE dynamo_quic_response_handshakes_total counter\n",
            "dynamo_quic_response_handshakes_total {}\n",
            "dynamo_quic_response_handshake_failures_total {}\n",
            "# TYPE dynamo_quic_response_reconnects_total counter\n",
            "dynamo_quic_response_reconnects_total {}\n",
            "dynamo_quic_response_connection_failures_total {}\n",
            "# TYPE dynamo_quic_response_stream_open_stalls_total counter\n",
            "dynamo_quic_response_stream_open_stalls_total {}\n",
            "# TYPE dynamo_quic_response_payloads_total counter\n",
            "dynamo_quic_response_payloads_total {}\n",
            "# TYPE dynamo_quic_response_resets_total counter\n",
            "dynamo_quic_response_resets_total{{reason=\"malformed\"}} {}\n",
            "dynamo_quic_response_resets_total{{reason=\"unknown_registration\"}} {}\n",
            "dynamo_quic_response_resets_total{{reason=\"receiver_dropped\"}} {}\n",
            "dynamo_quic_response_resets_total{{reason=\"publisher_dropped\"}} {}\n",
            "dynamo_quic_response_resets_total{{reason=\"oversized_frame\"}} {}\n",
            "dynamo_quic_response_resets_total{{reason=\"killed\"}} {}\n",
            "# TYPE dynamo_quic_response_udp_tx_bytes_total counter\n",
            "dynamo_quic_response_udp_tx_bytes_total {}\n",
            "dynamo_quic_response_udp_tx_datagrams_total {}\n",
            "dynamo_quic_response_udp_tx_ios_total {}\n",
            "dynamo_quic_response_udp_rx_bytes_total {}\n",
            "dynamo_quic_response_udp_rx_datagrams_total {}\n",
            "dynamo_quic_response_udp_rx_ios_total {}\n",
            "dynamo_quic_response_stream_frames_tx_total {}\n",
            "dynamo_quic_response_connection_blocked_frames_tx_total {}\n",
            "dynamo_quic_response_stream_blocked_frames_tx_total {}\n",
            "dynamo_quic_response_lost_packets_total {}\n",
            "dynamo_quic_response_lost_bytes_total {}\n",
            "# TYPE dynamo_quic_response_rtt_microseconds gauge\n",
            "dynamo_quic_response_rtt_microseconds {}\n",
            "dynamo_quic_response_mtu_bytes {}\n",
            "dynamo_quic_response_payloads_per_udp_datagram {}\n",
            "dynamo_quic_response_stream_frames_per_udp_datagram {}\n",
            "dynamo_quic_response_udp_datagrams_per_io {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"client\",direction=\"send\",kind=\"requested\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"client\",direction=\"send\",kind=\"actual\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"client\",direction=\"receive\",kind=\"requested\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"client\",direction=\"receive\",kind=\"actual\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"server\",direction=\"send\",kind=\"requested\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"server\",direction=\"send\",kind=\"actual\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"server\",direction=\"receive\",kind=\"requested\"}} {}\n",
            "dynamo_quic_response_udp_socket_buffer_bytes{{role=\"server\",direction=\"receive\",kind=\"actual\"}} {}\n",
        ),
        counters.active_connections,
        counters.active_streams,
        counters.handshakes,
        counters.handshake_failures,
        counters.reconnects,
        counters.connection_failures,
        counters.stream_open_stalls,
        counters.response_payloads,
        counters.malformed_resets,
        counters.unknown_registration_resets,
        counters.receiver_dropped_resets,
        counters.publisher_dropped_resets,
        counters.oversized_frame_resets,
        counters.killed_resets,
        sum(|stats| stats.udp_tx_bytes) + closed(&CLOSED_CONNECTION_COUNTERS.udp_tx_bytes),
        udp_tx_datagrams,
        udp_tx_ios,
        sum(|stats| stats.udp_rx_bytes) + closed(&CLOSED_CONNECTION_COUNTERS.udp_rx_bytes),
        sum(|stats| stats.udp_rx_datagrams) + closed(&CLOSED_CONNECTION_COUNTERS.udp_rx_datagrams),
        sum(|stats| stats.udp_rx_ios) + closed(&CLOSED_CONNECTION_COUNTERS.udp_rx_ios),
        stream_frames,
        sum(|stats| stats.connection_blocked_frames_tx)
            + closed(&CLOSED_CONNECTION_COUNTERS.connection_blocked_frames_tx),
        sum(|stats| stats.stream_blocked_frames_tx)
            + closed(&CLOSED_CONNECTION_COUNTERS.stream_blocked_frames_tx),
        sum(|stats| stats.lost_packets) + closed(&CLOSED_CONNECTION_COUNTERS.lost_packets),
        sum(|stats| stats.lost_bytes) + closed(&CLOSED_CONNECTION_COUNTERS.lost_bytes),
        rtt_micros,
        mtu,
        packet_density,
        stream_frame_density,
        datagrams_per_io,
        UDP_SOCKET_BUFFER,
        CLIENT_UDP_SEND_BUFFER.load(Ordering::Relaxed),
        UDP_SOCKET_BUFFER,
        CLIENT_UDP_RECV_BUFFER.load(Ordering::Relaxed),
        UDP_SOCKET_BUFFER,
        SERVER_UDP_SEND_BUFFER.load(Ordering::Relaxed),
        UDP_SOCKET_BUFFER,
        SERVER_UDP_RECV_BUFFER.load(Ordering::Relaxed),
    )
}

pub async fn create_response_stream(
    context: Arc<dyn AsyncEngineContext>,
    info: ConnectionInfo,
    cancellation_counter: Option<prometheus::IntCounter>,
) -> Result<ResponseStreamSender> {
    create_response_stream_in_pool(client_pool(), context, info, cancellation_counter).await
}

async fn create_response_stream_in_pool(
    pool: &ClientPool,
    context: Arc<dyn AsyncEngineContext>,
    info: ConnectionInfo,
    cancellation_counter: Option<prometheus::IntCounter>,
) -> Result<ResponseStreamSender> {
    let info = QuicResponseConnectionInfo::try_from(info)?;
    if info.context != context.id() {
        bail!(
            "QUIC response context mismatch: connection has {}, request has {}",
            info.context,
            context.id()
        );
    }

    for attempt in 0..2 {
        let connection = pool.connection(&info).await?;
        let opened = time::timeout(OPEN_TIMEOUT, connection.open_bi()).await;
        let (send, recv) = match opened {
            Ok(Ok(streams)) => streams,
            Ok(Err(error)) if attempt == 0 => {
                tracing::warn!(%error, "cached QUIC connection could not open a stream; reconnecting");
                COUNTERS.reconnects.fetch_add(1, Ordering::Relaxed);
                pool.invalidate(&info);
                continue;
            }
            Ok(Err(error)) => return Err(error.into()),
            Err(_) => {
                COUNTERS.stream_open_stalls.fetch_add(1, Ordering::Relaxed);
                COUNTERS.reconnects.fetch_add(1, Ordering::Relaxed);
                pool.invalidate(&info);
                if attempt == 0 {
                    continue;
                }
                bail!("timed out opening QUIC response stream");
            }
        };

        COUNTERS.active_streams.fetch_add(1, Ordering::Relaxed);
        spawn_control_reader(recv, context, cancellation_counter);
        return match ResponseStreamSender::new(send, info.response_registration_id).await {
            Ok(sender) => Ok(sender),
            Err(error) => {
                COUNTERS.active_streams.fetch_sub(1, Ordering::Relaxed);
                Err(error)
            }
        };
    }
    unreachable!("QUIC stream-open retry loop returns on both attempts")
}

fn spawn_control_reader(
    mut recv: RecvStream,
    context: Arc<dyn AsyncEngineContext>,
    cancellation_counter: Option<prometheus::IntCounter>,
) {
    tokio::spawn(async move {
        loop {
            match recv.read_u8().await {
                Ok(CONTROL_STOP) => {
                    if let Some(counter) = &cancellation_counter {
                        counter.inc();
                    }
                    context.stop();
                }
                Ok(CONTROL_KILL) => {
                    if let Some(counter) = &cancellation_counter {
                        counter.inc();
                    }
                    context.kill();
                    break;
                }
                Ok(value) => {
                    tracing::warn!(value, "malformed QUIC response control");
                    COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                    context.kill();
                    let _ = recv.stop(reset_code(RESET_MALFORMED));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => {
                    tracing::debug!(%error, "QUIC response control stream ended");
                    if !context.is_stopped() {
                        context.kill();
                    }
                    break;
                }
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
pub struct ServerOptions {
    pub address: IpAddr,
    pub port: u16,
}

struct RequestedResponse {
    context: Arc<dyn AsyncEngineContext>,
    connection: oneshot::Sender<Result<StreamReceiver, String>>,
    send_buffer_count: usize,
}

#[derive(Default)]
struct ServerState {
    pending: HashMap<Uuid, RequestedResponse>,
    registration_instance: HashMap<Uuid, EndpointInstanceId>,
    instance_registrations: HashMap<EndpointInstanceId, HashSet<Uuid>>,
    removed_instances: HashMap<EndpointInstanceId, time::Instant>,
}

pub struct QuicStreamServer {
    local_addr: SocketAddr,
    frontend_server_id: Uuid,
    certificate_fingerprint: [u8; 32],
    state: Arc<Mutex<ServerState>>,
    _endpoint: Endpoint,
}

impl QuicStreamServer {
    pub async fn new(options: ServerOptions) -> Result<Arc<Self>> {
        let identity = process_identity()?;
        let cert_der = CertificateDer::from(identity.certificate.clone());
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.private_key.clone()));

        let mut tls = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key)?;
        tls.alpn_protocols = vec![ALPN.to_vec()];
        tls.max_early_data_size = 0;
        let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls)?;
        let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(crypto));
        server_config.transport_config(Arc::new(server_transport_config()?));

        let bind = SocketAddr::new(options.address, options.port);
        let (socket, send_buffer, recv_buffer) = udp_socket(bind)?;
        SERVER_UDP_SEND_BUFFER.store(send_buffer as u64, Ordering::Relaxed);
        SERVER_UDP_RECV_BUFFER.store(recv_buffer as u64, Ordering::Relaxed);
        let local_addr = socket.local_addr()?;
        tracing::info!(
            address = %local_addr,
            requested_send_buffer = UDP_SOCKET_BUFFER,
            actual_send_buffer = send_buffer,
            requested_recv_buffer = UDP_SOCKET_BUFFER,
            actual_recv_buffer = recv_buffer,
            "configured QUIC response server UDP socket"
        );
        let endpoint = Endpoint::new(
            quinn::EndpointConfig::default(),
            Some(server_config),
            socket,
            Arc::new(quinn::TokioRuntime),
        )?;
        let state = Arc::new(Mutex::new(ServerState::default()));
        tokio::spawn(accept_connections(endpoint.clone(), state.clone()));

        Ok(Arc::new(Self {
            local_addr,
            frontend_server_id: identity.frontend_server_id,
            certificate_fingerprint: identity.certificate_fingerprint,
            state,
            _endpoint: endpoint,
        }))
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub async fn register_response(
        self: &Arc<Self>,
        options: StreamOptions,
    ) -> RegisteredStream<StreamReceiver> {
        assert!(options.enable_response_stream);
        let registration_id = Uuid::new_v4();
        let context_id = options.context.id().to_string();
        let (tx, rx) = oneshot::channel();
        self.state.lock().pending.insert(
            registration_id,
            RequestedResponse {
                context: options.context,
                connection: tx,
                send_buffer_count: options.send_buffer_count,
            },
        );
        let info = QuicResponseConnectionInfo {
            protocol_version: PROTOCOL_VERSION,
            address: self.local_addr.to_string(),
            frontend_server_id: self.frontend_server_id,
            response_registration_id: registration_id,
            context: context_id,
            certificate_fingerprint: self.certificate_fingerprint,
        };
        let state = self.state.clone();
        RegisteredStream::new(info.into(), rx).with_cleanup(move || {
            remove_registration(&state, registration_id);
        })
    }

    pub async fn associate_instance(&self, registration_id: Uuid, id: &EndpointInstanceId) -> bool {
        let mut state = self.state.lock();
        prune_tombstones(&mut state.removed_instances);
        if state.removed_instances.contains_key(id) {
            state.pending.remove(&registration_id);
            return false;
        }
        state
            .registration_instance
            .insert(registration_id, id.clone());
        state
            .instance_registrations
            .entry(id.clone())
            .or_default()
            .insert(registration_id);
        true
    }

    pub async fn cancel_response_stream(&self, registration_id: Uuid) {
        remove_registration(&self.state, registration_id);
    }

    pub async fn cancel_instance_streams(&self, id: &EndpointInstanceId) -> usize {
        let mut state = self.state.lock();
        prune_tombstones(&mut state.removed_instances);
        state
            .removed_instances
            .insert(id.clone(), time::Instant::now());
        let registrations = state.instance_registrations.remove(id).unwrap_or_default();
        for registration_id in &registrations {
            state.pending.remove(registration_id);
            state.registration_instance.remove(registration_id);
        }
        registrations.len()
    }

    pub async fn clear_instance_tombstone(&self, id: &EndpointInstanceId) {
        self.state.lock().removed_instances.remove(id);
    }
}

fn remove_registration(state: &Mutex<ServerState>, registration_id: Uuid) {
    let mut state = state.lock();
    state.pending.remove(&registration_id);
    if let Some(instance) = state.registration_instance.remove(&registration_id)
        && let Some(registrations) = state.instance_registrations.get_mut(&instance)
    {
        registrations.remove(&registration_id);
        if registrations.is_empty() {
            state.instance_registrations.remove(&instance);
        }
    }
}

fn take_registration(
    state: &Mutex<ServerState>,
    registration_id: Uuid,
) -> Option<RequestedResponse> {
    let mut state = state.lock();
    let pending = state.pending.remove(&registration_id);
    if let Some(instance) = state.registration_instance.remove(&registration_id)
        && let Some(registrations) = state.instance_registrations.get_mut(&instance)
    {
        registrations.remove(&registration_id);
        if registrations.is_empty() {
            state.instance_registrations.remove(&instance);
        }
    }
    pending
}

fn prune_tombstones(tombstones: &mut HashMap<EndpointInstanceId, time::Instant>) {
    let now = time::Instant::now();
    tombstones.retain(|_, inserted| now.saturating_duration_since(*inserted) < TOMBSTONE_TTL);
}

async fn accept_connections(endpoint: Endpoint, state: Arc<Mutex<ServerState>>) {
    while let Some(incoming) = endpoint.accept().await {
        let state = state.clone();
        tokio::spawn(async move {
            match incoming.await {
                Ok(connection) => {
                    COUNTERS.handshakes.fetch_add(1, Ordering::Relaxed);
                    COUNTERS.active_connections.fetch_add(1, Ordering::Relaxed);
                    while let Ok((send, recv)) = connection.accept_bi().await {
                        let state = state.clone();
                        COUNTERS.active_streams.fetch_add(1, Ordering::Relaxed);
                        tokio::spawn(async move {
                            if let Err(error) = process_response_stream(send, recv, state).await {
                                tracing::debug!(%error, "QUIC response stream failed");
                            }
                            COUNTERS.active_streams.fetch_sub(1, Ordering::Relaxed);
                        });
                    }
                    COUNTERS.connection_failures.fetch_add(1, Ordering::Relaxed);
                    COUNTERS.active_connections.fetch_sub(1, Ordering::Relaxed);
                }
                Err(error) => {
                    COUNTERS.handshake_failures.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(%error, "QUIC response handshake failed");
                }
            }
        });
    }
}

async fn process_response_stream(
    mut control: SendStream,
    recv: RecvStream,
    state: Arc<Mutex<ServerState>>,
) -> Result<()> {
    let mut reader = ChunkReader::new(recv);
    let registration_bytes = match reader.read_bytes(16).await {
        Ok(bytes) => bytes,
        Err(error) => {
            COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
            reader.stop(RESET_MALFORMED);
            let _ = control.reset(reset_code(RESET_MALFORMED));
            return Err(error);
        }
    };
    let registration_id = Uuid::from_u128(u128::from_be_bytes(
        registration_bytes
            .as_ref()
            .try_into()
            .expect("registration frame is exactly 16 bytes"),
    ));
    let Some(requested) = take_registration(&state, registration_id) else {
        COUNTERS
            .unknown_registration_resets
            .fetch_add(1, Ordering::Relaxed);
        reader.stop(RESET_UNKNOWN_REGISTRATION);
        let _ = control.reset(reset_code(RESET_UNKNOWN_REGISTRATION));
        bail!("unknown or duplicate QUIC response registration {registration_id}");
    };

    let prologue = match reader.read_bytes(5).await {
        Ok(prologue) => prologue,
        Err(error) => {
            COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
            reader.stop(RESET_MALFORMED);
            let _ = control.reset(reset_code(RESET_MALFORMED));
            let _ = requested
                .connection
                .send(Err("QUIC response ended before its prologue".to_string()));
            return Err(error);
        }
    };
    let status = prologue[0];
    let error_len = u32::from_be_bytes(prologue[1..5].try_into().unwrap()) as usize;
    if error_len > MAX_ERROR_BYTES {
        COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
        reader.stop(RESET_MALFORMED);
        let _ = control.reset(reset_code(RESET_MALFORMED));
        let _ = requested.connection.send(Err(format!(
            "QUIC response error prologue exceeds {MAX_ERROR_BYTES} bytes"
        )));
        return Ok(());
    }
    let error = match reader.read_bytes(error_len).await {
        Ok(error) => error,
        Err(read_error) => {
            COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
            reader.stop(RESET_MALFORMED);
            let _ = control.reset(reset_code(RESET_MALFORMED));
            let _ = requested.connection.send(Err(
                "QUIC response ended inside its error prologue".to_string()
            ));
            return Err(read_error);
        }
    };
    match status {
        STATUS_ERROR => {
            let error = match String::from_utf8(error.to_vec()) {
                Ok(error) => error,
                Err(_) => {
                    COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                    reader.stop(RESET_MALFORMED);
                    let _ = control.reset(reset_code(RESET_MALFORMED));
                    let _ = requested
                        .connection
                        .send(Err("QUIC response error is not UTF-8".to_string()));
                    return Ok(());
                }
            };
            match reader.read_optional_u32().await {
                Ok(None) => {}
                Ok(Some(_)) => {
                    COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                    reader.stop(RESET_MALFORMED);
                    let _ = control.reset(reset_code(RESET_MALFORMED));
                    let _ = requested.connection.send(Err(
                        "QUIC error response carried a payload frame".to_string(),
                    ));
                    return Ok(());
                }
                Err(read_error) => {
                    COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                    reader.stop(RESET_MALFORMED);
                    let _ = control.reset(reset_code(RESET_MALFORMED));
                    let _ = requested.connection.send(Err(
                        "QUIC error response ended with a partial frame".to_string(),
                    ));
                    return Err(read_error);
                }
            }
            let _ = requested.connection.send(Err(error));
            let _ = control.finish();
            return Ok(());
        }
        STATUS_SUCCESS if error_len == 0 => {}
        _ => {
            COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
            reader.stop(RESET_MALFORMED);
            let _ = control.reset(reset_code(RESET_MALFORMED));
            let _ = requested
                .connection
                .send(Err("malformed QUIC response prologue".to_string()));
            return Ok(());
        }
    }

    let (payload_tx, payload_rx) = tokio::sync::mpsc::channel(requested.send_buffer_count.max(1));
    if requested
        .connection
        .send(Ok(StreamReceiver { rx: payload_rx }))
        .is_err()
    {
        COUNTERS
            .receiver_dropped_resets
            .fetch_add(1, Ordering::Relaxed);
        reader.stop(RESET_RECEIVER_DROPPED);
        let _ = control.reset(reset_code(RESET_RECEIVER_DROPPED));
        return Ok(());
    }

    let context = requested.context;
    enum ControlClose {
        Finish,
        Reset(u32),
    }
    let (control_close_tx, mut control_close_rx) = oneshot::channel();
    let control_context = context.clone();
    let control_task = tokio::spawn(async move {
        tokio::select! {
            biased;
            close = &mut control_close_rx => {
                match close.unwrap_or(ControlClose::Reset(RESET_MALFORMED)) {
                    ControlClose::Finish => { let _ = control.finish(); }
                    ControlClose::Reset(code) => { let _ = control.reset(reset_code(code)); }
                }
            }
            _ = control_context.killed() => {
                let _ = control.write_all(&[CONTROL_KILL]).await;
                let _ = control.finish();
            }
            _ = control_context.stopped() => {
                if control.write_all(&[CONTROL_STOP]).await.is_ok() {
                    // Stop is graceful. Keep the reverse stream alive so a later
                    // hard kill can still be delivered while final output drains.
                    tokio::select! {
                        _ = control_context.killed() => {
                            let _ = control.write_all(&[CONTROL_KILL]).await;
                        }
                        close = &mut control_close_rx => {
                            match close.unwrap_or(ControlClose::Reset(RESET_MALFORMED)) {
                                ControlClose::Finish => {}
                                ControlClose::Reset(code) => {
                                    let _ = control.reset(reset_code(code));
                                    return;
                                }
                            }
                        }
                    }
                }
                let _ = control.finish();
            }
        }
    });

    let (close, result) = loop {
        tokio::select! {
            biased;
            _ = context.killed() => {
                COUNTERS.killed_resets.fetch_add(1, Ordering::Relaxed);
                reader.stop(RESET_KILLED);
                break (ControlClose::Reset(RESET_KILLED), Ok(()));
            }
            _ = payload_tx.closed() => {
                COUNTERS.receiver_dropped_resets.fetch_add(1, Ordering::Relaxed);
                reader.stop(RESET_RECEIVER_DROPPED);
                break (ControlClose::Reset(RESET_RECEIVER_DROPPED), Ok(()));
            }
            length = reader.read_optional_u32() => {
                let length = match length {
                    Ok(Some(length)) => length,
                    Ok(None) => break (ControlClose::Finish, Ok(())),
                    Err(error) => {
                        COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                        reader.stop(RESET_MALFORMED);
                        break (ControlClose::Reset(RESET_MALFORMED), Err(error));
                    }
                };
                let length = length as usize;
                if let Err(error) = validate_payload_size(length) {
                    COUNTERS.oversized_frame_resets.fetch_add(1, Ordering::Relaxed);
                    reader.stop(RESET_OVERSIZED_FRAME);
                    break (
                        ControlClose::Reset(RESET_OVERSIZED_FRAME),
                        Err(error),
                    );
                }
                let payload = match reader.read_bytes(length).await {
                    Ok(payload) => payload,
                    Err(error) => {
                        COUNTERS.malformed_resets.fetch_add(1, Ordering::Relaxed);
                        reader.stop(RESET_MALFORMED);
                        break (ControlClose::Reset(RESET_MALFORMED), Err(error));
                    }
                };
                if payload_tx.send(payload).await.is_err() {
                    COUNTERS.receiver_dropped_resets.fetch_add(1, Ordering::Relaxed);
                    reader.stop(RESET_RECEIVER_DROPPED);
                    break (ControlClose::Reset(RESET_RECEIVER_DROPPED), Ok(()));
                }
            }
        }
    };
    let _ = control_close_tx.send(close);
    let _ = control_task.await;
    result
}

/// Reads ordered QUIC stream chunks without copying contiguous payloads.
/// Headers are tiny and copied into fixed storage; a payload is coalesced only
/// when Quinn delivered it across multiple chunks.
struct ChunkReader {
    recv: RecvStream,
    chunks: VecDeque<Bytes>,
    finished: bool,
}

impl ChunkReader {
    fn new(recv: RecvStream) -> Self {
        Self {
            recv,
            chunks: VecDeque::new(),
            finished: false,
        }
    }

    fn stop(&mut self, code: u32) {
        let _ = self.recv.stop(reset_code(code));
    }

    async fn fill(&mut self) -> Result<bool> {
        if !self.chunks.is_empty() {
            return Ok(true);
        }
        if self.finished {
            return Ok(false);
        }
        match self.recv.read_chunk(64 * 1024, true).await? {
            Some(chunk) => {
                self.chunks.push_back(chunk.bytes);
                Ok(true)
            }
            None => {
                self.finished = true;
                Ok(false)
            }
        }
    }

    async fn read_optional_u32(&mut self) -> Result<Option<u32>> {
        if !self.fill().await? {
            return Ok(None);
        }
        let bytes = self.read_bytes(4).await?;
        Ok(Some(u32::from_be_bytes(bytes.as_ref().try_into().unwrap())))
    }

    async fn read_bytes(&mut self, len: usize) -> Result<Bytes> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        if !self.fill().await? {
            bail!("QUIC response stream ended with a partial frame");
        }
        if self.chunks.front().is_some_and(|chunk| chunk.len() >= len) {
            let chunk = self.chunks.front_mut().unwrap();
            let result = chunk.split_to(len);
            if chunk.is_empty() {
                self.chunks.pop_front();
            }
            return Ok(result);
        }

        let mut remaining = len;
        let mut result = BytesMut::with_capacity(len);
        while remaining != 0 {
            if !self.fill().await? {
                bail!("QUIC response stream ended with a partial frame");
            }
            let available = self.chunks.front().unwrap().len().min(remaining);
            let chunk = self.chunks.front_mut().unwrap().split_to(available);
            result.extend_from_slice(&chunk);
            remaining -= available;
            if self.chunks.front().unwrap().is_empty() {
                self.chunks.pop_front();
            }
        }
        Ok(result.freeze())
    }
}

fn server_transport_config() -> Result<TransportConfig> {
    let (stream_receive_window, connection_window, send_window) = flow_control_windows();
    let mut transport = TransportConfig::default();
    transport
        .max_concurrent_bidi_streams(VarInt::from_u32(MAX_BIDI_STREAMS))
        .max_concurrent_uni_streams(VarInt::from_u32(0))
        .stream_receive_window(VarInt::from_u32(stream_receive_window))
        .receive_window(VarInt::from_u32(connection_window))
        .send_window(send_window)
        .send_fairness(true)
        .mtu_discovery_config(Some(Default::default()))
        .enable_segmentation_offload(true)
        .max_idle_timeout(Some(IDLE_TIMEOUT.try_into()?));
    Ok(transport)
}

fn client_transport_config(qlog: Option<quinn::QlogStream>) -> Result<TransportConfig> {
    let (_, connection_window, send_window) = flow_control_windows();
    let mut transport = TransportConfig::default();
    transport
        .max_concurrent_bidi_streams(VarInt::from_u32(MAX_BIDI_STREAMS))
        .max_concurrent_uni_streams(VarInt::from_u32(0))
        .receive_window(VarInt::from_u32(connection_window))
        .send_window(send_window)
        .send_fairness(true)
        .mtu_discovery_config(Some(Default::default()))
        .enable_segmentation_offload(true)
        .max_idle_timeout(Some(IDLE_TIMEOUT.try_into()?));
    if let Some(stream) = qlog.or(diagnostic_qlog_stream()?) {
        transport.qlog_stream(Some(stream));
    }
    Ok(transport)
}

fn flow_control_windows() -> (u32, u32, u64) {
    if crate::config::env_is_truthy(
        crate::config::environment_names::tcp_response_stream::DYN_RESPONSE_STREAM_HIGH_WINDOW,
    ) {
        (
            HIGH_STREAM_RECEIVE_WINDOW,
            HIGH_CONNECTION_WINDOW,
            HIGH_SEND_WINDOW,
        )
    } else {
        (STREAM_RECEIVE_WINDOW, CONNECTION_WINDOW, SEND_WINDOW)
    }
}

fn diagnostic_qlog_stream() -> Result<Option<quinn::QlogStream>> {
    let Ok(base_path) = std::env::var(
        crate::config::environment_names::tcp_response_stream::DYN_RESPONSE_QLOG_PATH,
    ) else {
        return Ok(None);
    };
    if base_path.is_empty() {
        return Ok(None);
    }
    let sequence = QLOG_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path = format!("{base_path}.{}.{}.sqlog", std::process::id(), sequence);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create QUIC qlog at {path}"))?;
    let mut config = quinn::QlogConfig::default();
    config
        .writer(Box::new(file))
        .title(Some("Dynamo QUIC response transport".to_string()))
        .description(Some(
            "Diagnostic capture; disable DYN_RESPONSE_QLOG_PATH for performance measurements"
                .to_string(),
        ));
    let stream = config.into_stream();
    tracing::info!(%path, "enabled diagnostic QUIC qlog capture");
    Ok(stream)
}

fn client_config(
    fingerprint: [u8; 32],
    qlog: Option<quinn::QlogStream>,
) -> Result<quinn::ClientConfig> {
    let verifier = Arc::new(FingerprintVerifier {
        fingerprint,
        provider: Arc::new(rustls::crypto::ring::default_provider()),
    });
    let mut tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .dangerous()
    .with_custom_certificate_verifier(verifier)
    .with_no_client_auth();
    tls.alpn_protocols = vec![ALPN.to_vec()];
    tls.enable_early_data = false;
    let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls)?;
    let mut config = quinn::ClientConfig::new(Arc::new(crypto));
    config.transport_config(Arc::new(client_transport_config(qlog)?));
    Ok(config)
}

#[derive(Debug)]
struct FingerprintVerifier {
    fingerprint: [u8; 32],
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::client::danger::ServerCertVerifier for FingerprintVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let actual: [u8; 32] = Sha256::digest(end_entity.as_ref()).into();
        if actual != self.fingerprint {
            return Err(rustls::Error::General(
                "QUIC response certificate fingerprint mismatch".to_string(),
            ));
        }
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
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
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn udp_socket(bind: SocketAddr) -> Result<(std::net::UdpSocket, usize, usize)> {
    let domain = if bind.is_ipv6() {
        Domain::IPV6
    } else {
        Domain::IPV4
    };
    let socket = Socket::new(domain, Type::DGRAM, Some(Protocol::UDP))?;
    if bind.is_ipv6() {
        socket.set_only_v6(true)?;
    }
    socket.set_reuse_address(true)?;
    socket.set_send_buffer_size(UDP_SOCKET_BUFFER)?;
    socket.set_recv_buffer_size(UDP_SOCKET_BUFFER)?;
    let actual_send_buffer = socket.send_buffer_size()?;
    let actual_recv_buffer = socket.recv_buffer_size()?;
    socket.bind(&bind.into())?;
    socket.set_nonblocking(true)?;
    Ok((socket.into(), actual_send_buffer, actual_recv_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::AsyncEngineContextProvider, pipeline::Context};
    use futures::future::join_all;

    #[test]
    fn connection_info_rejects_wrong_version() {
        let info = QuicResponseConnectionInfo {
            protocol_version: PROTOCOL_VERSION + 1,
            address: "127.0.0.1:1234".to_string(),
            frontend_server_id: Uuid::new_v4(),
            response_registration_id: Uuid::new_v4(),
            context: "request".to_string(),
            certificate_fingerprint: [1; 32],
        };
        assert!(info.validate().is_err());

        let valid = QuicResponseConnectionInfo {
            protocol_version: PROTOCOL_VERSION,
            address: "127.0.0.1:1234".to_string(),
            frontend_server_id: Uuid::new_v4(),
            response_registration_id: Uuid::new_v4(),
            context: "request".to_string(),
            certificate_fingerprint: [1; 32],
        };
        assert!(valid.validate().is_ok());
        for invalid in [
            QuicResponseConnectionInfo {
                address: "not-a-socket".to_string(),
                ..valid.clone()
            },
            QuicResponseConnectionInfo {
                frontend_server_id: Uuid::nil(),
                ..valid.clone()
            },
            QuicResponseConnectionInfo {
                response_registration_id: Uuid::nil(),
                ..valid.clone()
            },
            QuicResponseConnectionInfo {
                context: String::new(),
                ..valid.clone()
            },
            QuicResponseConnectionInfo {
                certificate_fingerprint: [0; 32],
                ..valid
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[tokio::test]
    async fn error_prologue_and_size_limit_are_enforced() {
        let max_payload =
            crate::pipeline::network::get_tcp_max_message_size().min(u32::MAX as usize);
        assert!(validate_payload_size(max_payload).is_ok());
        assert!(validate_payload_size(max_payload + 1).is_err());

        let pool = ClientPool::new();
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();

        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        sender
            .send_prologue(Some("worker failed".to_string()))
            .await
            .unwrap();
        assert!(
            sender
                .send(Bytes::from_static(b"invalid payload"))
                .await
                .is_err()
        );
        sender.finish().unwrap();
        let (_, provider) = registered.into_parts();
        match provider.await.unwrap() {
            Ok(_) => panic!("error prologue produced a response stream"),
            Err(error) => assert_eq!(error, "worker failed"),
        }

        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info,
            None,
        )
        .await
        .unwrap();
        let error = "x".repeat(MAX_ERROR_BYTES + 1);
        assert!(sender.send_prologue(Some(error)).await.is_err());
    }

    #[tokio::test]
    async fn chunk_reader_retains_contiguous_payload_and_coalesces_fragments() {
        // The transport integration tests cover RecvStream. Keep the framing
        // invariant independently testable with the exact queue operations the
        // decoder uses.
        let contiguous = Bytes::from_static(b"abcdef");
        let pointer = contiguous.as_ptr();
        let mut queue = VecDeque::from([contiguous]);
        let sliced = queue.front_mut().unwrap().split_to(3);
        assert_eq!(sliced.as_ptr(), pointer);
        assert_eq!(&sliced[..], b"abc");

        let mut coalesced = BytesMut::with_capacity(4);
        for chunk in [Bytes::from_static(b"ab"), Bytes::from_static(b"cd")] {
            coalesced.extend_from_slice(&chunk);
        }
        assert_eq!(&coalesced.freeze()[..], b"abcd");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn one_connection_carries_one_thousand_logical_responses() {
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();
        let frontend_id = server.frontend_server_id;
        let pool = Arc::new(ClientPool::new());

        let responses = (0u32..1000)
            .map(|sequence| {
                let server = server.clone();
                let pool = pool.clone();
                async move {
                    let upstream = Context::new(());
                    let registered = server
                        .register_response(
                            StreamOptions::builder()
                                .context(upstream.context())
                                .enable_request_stream(false)
                                .enable_response_stream(true)
                                .build()
                                .unwrap(),
                        )
                        .await;
                    let info = registered.connection_info.clone();
                    let downstream = Context::with_id_and_metadata(
                        (),
                        upstream.id().to_string(),
                        Default::default(),
                    );
                    let worker = tokio::spawn(async move {
                        let mut sender =
                            create_response_stream_in_pool(&pool, downstream.context(), info, None)
                                .await
                                .unwrap();
                        sender.send_prologue(None).await.unwrap();
                        sender
                            .send(Bytes::copy_from_slice(&sequence.to_be_bytes()))
                            .await
                            .unwrap();
                        sender
                            .send(Bytes::from_static(b"second-frame"))
                            .await
                            .unwrap();
                        sender.finish().unwrap();
                    });

                    let (_, provider) = registered.into_parts();
                    let mut receiver = provider.await.unwrap().unwrap();
                    assert_eq!(
                        receiver.rx.recv().await.unwrap(),
                        Bytes::copy_from_slice(&sequence.to_be_bytes())
                    );
                    assert_eq!(
                        receiver.rx.recv().await.unwrap(),
                        Bytes::from_static(b"second-frame")
                    );
                    assert!(receiver.rx.recv().await.is_none());
                    worker.await.unwrap();
                }
            })
            .collect::<Vec<_>>();
        join_all(responses).await;

        let matching_connections = pool
            .connections
            .iter()
            .filter(|entry| entry.key().frontend_server_id == frontend_id)
            .count();
        assert_eq!(matching_connections, 1);
    }

    #[tokio::test]
    async fn certificate_fingerprint_is_pinned() {
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();
        let pool = ClientPool::new();
        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let mut info =
            QuicResponseConnectionInfo::try_from(registered.connection_info.clone()).unwrap();
        info.certificate_fingerprint[0] ^= 0xff;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        assert!(
            create_response_stream_in_pool(&pool, downstream.context(), info.into(), None)
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sustained_small_response_burst_produces_multiple_stream_frames_per_datagram() {
        let pool = Arc::new(ClientPool::new());
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(129));
        let mut publishers = Vec::new();
        let mut consumers = Vec::new();

        for sequence in 0u32..128 {
            let upstream = Context::new(());
            let registered = server
                .register_response(
                    StreamOptions::builder()
                        .context(upstream.context())
                        .enable_request_stream(false)
                        .enable_response_stream(true)
                        .build()
                        .unwrap(),
                )
                .await;
            let downstream =
                Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
            let mut sender = create_response_stream_in_pool(
                &pool,
                downstream.context(),
                registered.connection_info.clone(),
                None,
            )
            .await
            .unwrap();
            sender.send_prologue(None).await.unwrap();
            let (_, provider) = registered.into_parts();
            consumers.push(tokio::spawn(async move {
                let mut receiver = provider.await.unwrap().unwrap();
                assert_eq!(
                    receiver.rx.recv().await.unwrap(),
                    Bytes::copy_from_slice(&sequence.to_be_bytes())
                );
                assert!(receiver.rx.recv().await.is_none());
            }));
            let barrier = barrier.clone();
            publishers.push(tokio::spawn(async move {
                barrier.wait().await;
                sender
                    .send(Bytes::copy_from_slice(&sequence.to_be_bytes()))
                    .await
                    .unwrap();
                sender.finish().unwrap();
            }));
        }

        barrier.wait().await;
        for result in join_all(publishers).await {
            result.unwrap();
        }
        for result in join_all(consumers).await {
            result.unwrap();
        }
        time::sleep(Duration::from_millis(50)).await;

        let connection = pool.connections.iter().next().unwrap();
        let stats = connection.value().stats();
        assert!(
            stats.frame_tx.stream > stats.udp_tx.datagrams,
            "sustained burst did not produce more STREAM frames ({}) than UDP datagrams ({})",
            stats.frame_tx.stream,
            stats.udp_tx.datagrams,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stalled_response_does_not_block_a_sibling_stream() {
        let pool = Arc::new(ClientPool::new());
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();

        let stalled_upstream = Context::new(());
        let stalled_registration = server
            .register_response(
                StreamOptions::builder()
                    .context(stalled_upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .send_buffer_count(1)
                    .build()
                    .unwrap(),
            )
            .await;
        let stalled_downstream = Context::with_id_and_metadata(
            (),
            stalled_upstream.id().to_string(),
            Default::default(),
        );
        let mut stalled_sender = create_response_stream_in_pool(
            &pool,
            stalled_downstream.context(),
            stalled_registration.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        stalled_sender.send_prologue(None).await.unwrap();
        let (_, stalled_provider) = stalled_registration.into_parts();
        let stalled_receiver = stalled_provider.await.unwrap().unwrap();
        let stalled_writer = tokio::spawn(async move {
            let payload = Bytes::from(vec![7u8; 64 * 1024]);
            loop {
                stalled_sender.send(payload.clone()).await.unwrap();
            }
        });

        let sibling_upstream = Context::new(());
        let sibling_registration = server
            .register_response(
                StreamOptions::builder()
                    .context(sibling_upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let sibling_downstream = Context::with_id_and_metadata(
            (),
            sibling_upstream.id().to_string(),
            Default::default(),
        );
        let sibling_info = sibling_registration.connection_info.clone();
        let sibling = async {
            let mut sender = create_response_stream_in_pool(
                &pool,
                sibling_downstream.context(),
                sibling_info,
                None,
            )
            .await
            .unwrap();
            sender.send_prologue(None).await.unwrap();
            sender.send(Bytes::from_static(b"sibling")).await.unwrap();
            sender.finish().unwrap();
            let (_, provider) = sibling_registration.into_parts();
            let mut receiver = provider.await.unwrap().unwrap();
            assert_eq!(
                receiver.rx.recv().await.unwrap(),
                Bytes::from_static(b"sibling")
            );
            assert!(receiver.rx.recv().await.is_none());
        };
        time::timeout(Duration::from_secs(2), sibling)
            .await
            .expect("stalled stream blocked its sibling");

        stalled_writer.abort();
        drop(stalled_receiver);
    }

    #[tokio::test]
    async fn stop_is_graceful_and_kill_is_forwarded_on_reverse_half() {
        let pool = ClientPool::new();
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();

        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        sender.send_prologue(None).await.unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        upstream.context().stop();
        time::timeout(Duration::from_secs(1), downstream.context().stopped())
            .await
            .expect("Stop was not forwarded");
        sender
            .send(Bytes::from_static(b"graceful-final"))
            .await
            .unwrap();
        sender.finish().unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"graceful-final")
        );

        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        sender.send_prologue(None).await.unwrap();
        let (_, provider) = registered.into_parts();
        let _receiver = provider.await.unwrap().unwrap();
        upstream.context().kill();
        time::timeout(Duration::from_secs(1), downstream.context().killed())
            .await
            .expect("Kill was not forwarded");
    }

    #[tokio::test]
    async fn unknown_registration_resets_only_that_stream() {
        let pool = ClientPool::new();
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();
        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let valid_info = registered.connection_info.clone();
        let mut bad_info = QuicResponseConnectionInfo::try_from(valid_info.clone()).unwrap();
        bad_info.response_registration_id = Uuid::new_v4();
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut bad_sender =
            create_response_stream_in_pool(&pool, downstream.context(), bad_info.into(), None)
                .await
                .unwrap();
        let mut reset_observed = bad_sender.send_prologue(None).await.is_err();
        if !reset_observed {
            for _ in 0..100 {
                if bad_sender.send(Bytes::from_static(b"bad")).await.is_err() {
                    reset_observed = true;
                    break;
                }
                tokio::task::yield_now().await;
            }
        }
        assert!(reset_observed, "unknown registration stream was not reset");

        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut valid_sender =
            create_response_stream_in_pool(&pool, downstream.context(), valid_info.clone(), None)
                .await
                .unwrap();
        valid_sender.send_prologue(None).await.unwrap();
        valid_sender
            .send(Bytes::from_static(b"valid"))
            .await
            .unwrap();
        valid_sender.finish().unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"valid")
        );

        let duplicate_context =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut duplicate_sender =
            create_response_stream_in_pool(&pool, duplicate_context.context(), valid_info, None)
                .await
                .unwrap();
        let mut duplicate_reset = duplicate_sender.send_prologue(None).await.is_err();
        if !duplicate_reset {
            for _ in 0..100 {
                if duplicate_sender
                    .send(Bytes::from_static(b"duplicate"))
                    .await
                    .is_err()
                {
                    duplicate_reset = true;
                    break;
                }
                tokio::task::yield_now().await;
            }
        }
        assert!(
            duplicate_reset,
            "duplicate registration stream was not reset"
        );
    }

    #[tokio::test]
    async fn publisher_reset_closes_receiver_and_connection_reopens() {
        let pool = ClientPool::new();
        let server = QuicStreamServer::new(ServerOptions {
            address: "127.0.0.1".parse().unwrap(),
            port: 0,
        })
        .await
        .unwrap();
        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        sender.send_prologue(None).await.unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        drop(sender);
        let after_reset = time::timeout(Duration::from_secs(1), receiver.rx.recv())
            .await
            .expect("publisher reset did not close receiver");
        assert!(after_reset.is_none(), "publisher reset emitted a payload");

        let old_connection = pool.connections.iter().next().unwrap().value().clone();
        old_connection.close(reset_code(RESET_KILLED), b"test connection loss");
        time::sleep(Duration::from_millis(20)).await;

        let upstream = Context::new(());
        let registered = server
            .register_response(
                StreamOptions::builder()
                    .context(upstream.context())
                    .enable_request_stream(false)
                    .enable_response_stream(true)
                    .build()
                    .unwrap(),
            )
            .await;
        let downstream =
            Context::with_id_and_metadata((), upstream.id().to_string(), Default::default());
        let mut sender = create_response_stream_in_pool(
            &pool,
            downstream.context(),
            registered.connection_info.clone(),
            None,
        )
        .await
        .unwrap();
        sender.send_prologue(None).await.unwrap();
        sender
            .send(Bytes::from_static(b"reconnected"))
            .await
            .unwrap();
        sender.finish().unwrap();
        let (_, provider) = registered.into_parts();
        let mut receiver = provider.await.unwrap().unwrap();
        assert_eq!(
            receiver.rx.recv().await.unwrap(),
            Bytes::from_static(b"reconnected")
        );
    }
}
