// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Uses ZMQ PUB/SUB pattern for one-way event broadcasting:
//! - Publishers bind to endpoints and broadcast events
//! - Subscribers connect to endpoints and receive events
//! - Topic-based filtering at socket level for efficiency
//!
//! ## Message Format
//!
//! ZMQ multipart message:
//! - Frame 0: Topic (string) - for ZMQ subscription filtering
//! - Frame 1: publisher_id (8 bytes, u64 big-endian) - for fast deduplication
//! - Frame 2: sequence (8 bytes, u64 big-endian) - for fast deduplication
//! - Frame 3: Binary frame (5-byte header + EventEnvelope payload)

use anyhow::{Context as _, Result, anyhow};
use async_stream::stream;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, Stream, StreamExt};
use once_cell::sync::OnceCell;
use std::ffi::OsStr;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use thiserror::Error;
use tmq::{
    AsZmqSocket, Context, Message, Multipart, SocketBuilder,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::{Mutex, broadcast, watch};
use tokio_util::task::AbortOnDropHandle;

/// Returns the process-wide shared ZMQ context.
///
/// libzmq spawns background I/O threads per `Context`, so all PUB/SUB sockets
/// share one. `zmq::Context` is reference-counted; clones drive the same context.
fn shared_zmq_context() -> Result<Context> {
    static CONTEXT: OnceCell<Context> = OnceCell::new();
    CONTEXT
        .get_or_try_init(|| {
            let value = std::env::var_os("DYN_ZMQ_IO_THREADS");
            configured_zmq_context(value.as_deref())
        })
        .cloned()
}

fn configured_zmq_context(value: Option<&OsStr>) -> Result<Context> {
    let io_threads = value
        .unwrap_or_else(|| OsStr::new("4"))
        .to_str()
        .context("DYN_ZMQ_IO_THREADS must be valid UTF-8")?
        .parse::<i32>()
        .context("DYN_ZMQ_IO_THREADS must be a positive integer")?;
    anyhow::ensure!(
        io_threads > 0,
        "DYN_ZMQ_IO_THREADS must be a positive integer"
    );
    let context = Context::new();
    // Configure the process-wide context before creating any PUB/SUB sockets.
    context
        .set_io_threads(io_threads)
        .context("failed to apply DYN_ZMQ_IO_THREADS to the event-plane ZMQ context")?;
    tracing::info!(io_threads, "Configured shared event-plane ZMQ context");
    Ok(context)
}

/// High Water Mark (HWM) for ZMQ sockets.
/// This controls the maximum number of messages that can be queued.
/// Default ZMQ HWM is 1000, which limits scalability.
const ZMQ_SNDHWM: i32 = 100_000; // Send buffer: 100K messages
const ZMQ_RCVHWM: i32 = 100_000; // Receive buffer: 100K messages
const ZMQ_SNDTIMEOUT_MS: i32 = 0; // Send timeout: fail fast under pressure
const ZMQ_RCVTIMEOUT_MS: i32 = 100; // Receive timeout: 100ms (avoids blocking forever)

/// Keeps a socket's monitor alive until event production has been disabled.
///
/// libzmq sends monitor events synchronously. Dropping its PAIR receiver while
/// monitoring remains enabled can block an I/O thread on a later disconnect.
pub struct MonitoredZmqSocket<S: AsZmqSocket> {
    socket: S,
    readiness: Option<ZmqConnectionReadiness>,
}

impl<S: AsZmqSocket> MonitoredZmqSocket<S> {
    fn unmonitored(socket: S) -> Self {
        Self {
            socket,
            readiness: None,
        }
    }

    fn monitored(socket: S, context: &Context) -> Result<Self> {
        let readiness = monitor_connection(socket.get_socket(), context)?;
        Ok(Self {
            socket,
            readiness: Some(readiness),
        })
    }

    fn readiness(&self) -> ZmqConnectionReadiness {
        self.readiness
            .as_ref()
            .expect("socket is monitored")
            .clone()
    }
}

impl<S: AsZmqSocket> AsZmqSocket for MonitoredZmqSocket<S> {
    fn get_socket(&self) -> &zmq::Socket {
        self.socket.get_socket()
    }
}

impl<S: AsZmqSocket> Deref for MonitoredZmqSocket<S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.socket
    }
}

impl<S: AsZmqSocket> DerefMut for MonitoredZmqSocket<S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.socket
    }
}

impl<S: AsZmqSocket + Stream + Unpin> Stream for MonitoredZmqSocket<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().socket).poll_next(cx)
    }
}

impl<S: AsZmqSocket> Drop for MonitoredZmqSocket<S> {
    fn drop(&mut self) {
        if self.readiness.is_some() {
            // rust-zmq cannot pass the null endpoint used to stop monitoring.
            // Replacing the monitor with a zero-event monitor disables event
            // production synchronously, before the old receiver is aborted.
            // A fresh endpoint avoids racing the old monitor's asynchronous close.
            let endpoint = format!("inproc://dynamo-zmq-disabled-{}", uuid::Uuid::new_v4());
            if let Err(error) = self.socket.get_socket().monitor(&endpoint, 0) {
                tracing::debug!(%error, "Unable to disable closing ZMQ socket monitor");
            }
        }
    }
}

/// Tracks completed ZMTP handshakes independently of the data receive loop.
///
/// The subscription is configured before connecting. A completed handshake means
/// the transport can forward events; PUB/SUB does not acknowledge the remote
/// publisher's processing of SUBSCRIBE and this is not a delivery guarantee.
#[derive(Clone)]
pub struct ZmqConnectionReadiness {
    connected: watch::Receiver<std::collections::HashSet<String>>,
    endpoints: Arc<parking_lot::Mutex<std::collections::HashMap<String, String>>>,
    // Keep the receiver connected even if Tokio cancels its reader first during
    // shutdown. The owning data socket can then disable monitoring safely.
    _monitor_socket: Arc<Mutex<tmq::pair::Pair>>,
    _monitor_task: Arc<AbortOnDropHandle<()>>,
}

impl ZmqConnectionReadiness {
    /// Wait for the endpoint's transport handshake. Callers own the startup
    /// deadline and cancellation policy; dropping this future is safe.
    pub async fn wait_connected(&self, endpoint: &str) -> Result<()> {
        let mut aliases = std::collections::HashSet::from([endpoint.to_string()]);
        if let Some(recorded) = self.endpoints.lock().get(endpoint) {
            aliases.insert(recorded.clone());
        }
        if let Some(address) = endpoint.strip_prefix("tcp://") {
            if let Ok(address) = address.parse::<std::net::SocketAddr>() {
                aliases.insert(format!("tcp://{address}"));
            } else {
                // libzmq may report the hostname on its initial connection and
                // the resolved IP after reconnecting. LAST_ENDPOINT alone does
                // not cover both spellings. Resolve only startup hostnames;
                // numeric TCP and IPC endpoints do not perform a DNS lookup.
                let addresses = tokio::net::lookup_host(address)
                    .await
                    .with_context(|| format!("Failed to resolve ZMQ endpoint {endpoint}"))?;
                aliases.extend(addresses.map(|address| format!("tcp://{address}")));
            }
        }
        let mut connected = self.connected.clone();
        loop {
            if !connected.borrow_and_update().is_disjoint(&aliases) {
                return Ok(());
            }
            connected
                .changed()
                .await
                .context("ZMQ connection monitor stopped")?;
        }
    }

    fn record_endpoint(&self, socket: &zmq::Socket, endpoint: &str) -> Result<()> {
        // Preserve libzmq's initial spelling; reconnects may use a resolved IP.
        let resolved = socket
            .get_last_endpoint()?
            .map_err(|_| anyhow!("ZMQ endpoint is not valid UTF-8"))?;
        self.endpoints.lock().insert(endpoint.to_string(), resolved);
        Ok(())
    }
}

/// Start a SUB socket with an observable connection handshake.
///
/// Socket construction remains nonblocking. Readiness is observed separately so
/// adding an unavailable peer does not block an existing grouped receive loop.
pub fn connect_subscriber_with_readiness(
    endpoint: &str,
    topic: &str,
    rcvhwm: i32,
) -> Result<(MonitoredZmqSocket<Subscribe>, ZmqConnectionReadiness)> {
    anyhow::ensure!(rcvhwm > 0, "ZMQ receive HWM must be greater than zero");
    anyhow::ensure!(
        endpoint.starts_with("tcp://") || endpoint.starts_with("ipc://"),
        "ZMQ connection readiness requires a TCP or IPC endpoint"
    );
    let context = shared_zmq_context()?;
    let socket = context
        .socket(zmq::SUB)
        .map_err(|error| map_socket_creation_error(error.into()))?;
    socket.set_rcvhwm(rcvhwm)?;
    socket.set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)?;
    socket.set_linger(0)?;
    socket.set_reconnect_ivl(100)?;
    socket.set_reconnect_ivl_max(5000)?;
    socket.set_tcp_keepalive(1)?;
    socket.set_subscribe(topic.as_bytes())?;

    let socket =
        <tmq::subscribe::SubscribeWithoutTopic as tmq::FromZmqSocket<_>>::from_zmq_socket(socket)?
            .subscribe(topic.as_bytes())?;
    let socket = MonitoredZmqSocket::monitored(socket, &context)?;
    let readiness = socket.readiness();
    socket.get_socket().connect(endpoint)?;
    readiness.record_endpoint(socket.get_socket(), endpoint)?;
    Ok((socket, readiness))
}

fn monitor_connection(socket: &zmq::Socket, context: &Context) -> Result<ZmqConnectionReadiness> {
    let monitor_endpoint = format!("inproc://dynamo-zmq-ready-{}", uuid::Uuid::new_v4());
    let events = zmq::SocketEvent::HANDSHAKE_SUCCEEDED.to_raw()
        | zmq::SocketEvent::DISCONNECTED.to_raw()
        | zmq::SocketEvent::MONITOR_STOPPED.to_raw();
    let monitor = tmq::pair(context)
        .set_linger(0)
        .connect(&monitor_endpoint)?;
    socket.monitor(&monitor_endpoint, i32::from(events))?;

    let monitor = Arc::new(Mutex::new(monitor));
    let reader = monitor.clone();
    let (connected_tx, connected) = watch::channel(std::collections::HashSet::new());
    let monitor_task = tokio::spawn(async move {
        let mut monitor = reader.lock().await;
        while let Some(Ok(frames)) = monitor.next().await {
            if frames.len() != 2 || frames[0].len() < 2 {
                continue;
            }
            let event = u16::from_ne_bytes([frames[0][0], frames[0][1]]);
            if event == zmq::SocketEvent::MONITOR_STOPPED.to_raw() {
                break;
            }
            let Ok(endpoint) = std::str::from_utf8(&frames[1]) else {
                continue;
            };
            connected_tx.send_modify(|connected| {
                if event == zmq::SocketEvent::HANDSHAKE_SUCCEEDED.to_raw() {
                    connected.insert(endpoint.to_string());
                } else if event == zmq::SocketEvent::DISCONNECTED.to_raw() {
                    connected.remove(endpoint);
                }
            });
        }
    });
    Ok(ZmqConnectionReadiness {
        connected,
        endpoints: Arc::default(),
        _monitor_socket: monitor,
        _monitor_task: Arc::new(AbortOnDropHandle::new(monitor_task)),
    })
}

const ZMQ_SOCKET_LIMIT_GUIDANCE: &str = "ZMQ could not create another socket. The process may have reached libzmq's ZMQ_MAX_SOCKETS limit or its file-descriptor limit. Reduce direct-ZMQ peers or raise the limit with `ulimit -n`";
const PROCESS_FD_LIMIT_GUIDANCE: &str = "The process reached its file-descriptor limit. Reduce open file descriptors or raise the limit with `ulimit -n`";

use super::codec::{Codec, MsgpackCodec};
use super::frame::Frame;
use super::transport::{EventTransportRx, EventTransportTx, WireStream};
use crate::discovery::EventTransportKind;

fn socket_limit_guidance(raw_errno: Option<i32>, guidance: &'static str) -> Option<&'static str> {
    (raw_errno == Some(libc::EMFILE)).then_some(guidance)
}

fn error_with_guidance(
    error: impl std::error::Error + Send + Sync + 'static,
    guidance: &str,
) -> anyhow::Error {
    let message = format!("{error}. {guidance}");
    anyhow::Error::new(error).context(message)
}

fn map_socket_creation_error(error: tmq::TmqError) -> anyhow::Error {
    let guidance = match &error {
        tmq::TmqError::Zmq(error) => {
            socket_limit_guidance(Some(error.to_raw()), ZMQ_SOCKET_LIMIT_GUIDANCE)
        }
        tmq::TmqError::Io(error) => {
            socket_limit_guidance(error.raw_os_error(), PROCESS_FD_LIMIT_GUIDANCE)
        }
        tmq::TmqError::InterruptedSend => None,
    };

    match guidance {
        Some(guidance) => error_with_guidance(error, guidance),
        None => error.into(),
    }
}

fn bind_tmq_socket<T>(builder: SocketBuilder<T>, endpoint: &str) -> Result<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder.bind(endpoint).map_err(map_socket_creation_error)
}

fn connect_tmq_socket<T>(builder: SocketBuilder<T>, endpoint: &str) -> Result<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder.connect(endpoint).map_err(map_socket_creation_error)
}

fn configure_publish_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder
        .set_sndhwm(ZMQ_SNDHWM)
        .set_sndtimeo(ZMQ_SNDTIMEOUT_MS)
}

fn configure_subscribe_builder<T>(builder: SocketBuilder<T>) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    configure_subscribe_builder_with_hwm(builder, ZMQ_RCVHWM)
}

fn configure_subscribe_builder_with_hwm<T>(
    builder: SocketBuilder<T>,
    rcvhwm: i32,
) -> SocketBuilder<T>
where
    T: tmq::FromZmqSocket<T>,
{
    builder.set_rcvhwm(rcvhwm).set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
}

/// Keeps a received ZMQ message alive for as long as any derived `Bytes` exists.
///
/// `Bytes::from_owner` obtains the message data pointer only after moving this
/// owner into stable storage, so this also supports libzmq's inline messages.
struct ZmqMessageOwner(Message);

impl AsRef<[u8]> for ZmqMessageOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// ZMQ PUB transport for publishing events.
pub struct ZmqPubTransport {
    socket: Arc<Mutex<MonitoredZmqSocket<Publish>>>,
    topic: String,
    readiness: Option<(ZmqConnectionReadiness, Vec<String>)>,
}

impl ZmqPubTransport {
    /// Create a new ZMQ publisher by binding to an endpoint.
    ///
    /// If the TCP port is 0, ZMQ allocates and reserves an ephemeral port
    /// on the publisher socket itself.
    ///
    /// Returns the transport and the actual bound endpoint.
    pub async fn bind(endpoint: &str, topic: &str) -> Result<(Self, String)> {
        let bind_endpoint = if endpoint.starts_with("tcp://") && endpoint.ends_with(":0") {
            format!("{}*", &endpoint[..endpoint.len() - 1])
        } else {
            endpoint.to_string()
        };

        let ctx = shared_zmq_context()?;
        let socket = bind_tmq_socket(configure_publish_builder(publish(&ctx)), &bind_endpoint)?;
        let actual_endpoint = socket
            .get_socket()
            .get_last_endpoint()
            .context("Failed to read bound ZMQ publisher endpoint")?
            .map_err(|_| anyhow!("Bound ZMQ publisher endpoint is not valid UTF-8"))?;

        tracing::info!(
            endpoint = %actual_endpoint,
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport bound with configured HWM"
        );

        Ok((
            Self {
                socket: Arc::new(Mutex::new(MonitoredZmqSocket::unmonitored(socket))),
                topic: topic.to_string(),
                readiness: None,
            },
            actual_endpoint,
        ))
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Connect to single broker XSUB endpoint (broker mode)
    pub async fn connect(xsub_endpoint: &str, topic: &str) -> Result<Self> {
        Self::connect_multiple(&[xsub_endpoint.to_string()], topic).await
    }

    /// Connect to multiple broker XSUB endpoints (HA mode)
    pub async fn connect_multiple(xsub_endpoints: &[String], topic: &str) -> Result<Self> {
        let mut endpoints = xsub_endpoints.iter();
        let Some(first_endpoint) = endpoints.next() else {
            anyhow::bail!("Cannot connect to zero endpoints");
        };

        let ctx = shared_zmq_context()?;
        let socket = ctx
            .socket(zmq::PUB)
            .map_err(|error| map_socket_creation_error(error.into()))?;
        socket.set_sndhwm(ZMQ_SNDHWM)?;
        socket.set_sndtimeo(ZMQ_SNDTIMEOUT_MS)?;
        let socket = MonitoredZmqSocket::monitored(
            <Publish as tmq::FromZmqSocket<_>>::from_zmq_socket(socket)?,
            &ctx,
        )?;
        let readiness = socket.readiness();
        socket.get_socket().connect(first_endpoint)?;
        readiness.record_endpoint(socket.get_socket(), first_endpoint)?;

        for endpoint in endpoints {
            socket.get_socket().connect(endpoint)?;
            readiness.record_endpoint(socket.get_socket(), endpoint)?;
            tracing::debug!(endpoint = %endpoint, "ZMQ PUB connected to broker XSUB");
        }

        tracing::info!(
            num_endpoints = xsub_endpoints.len(),
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport connected to multiple broker XSUBs with configured HWM"
        );

        Ok(Self {
            socket: Arc::new(Mutex::new(socket)),
            topic: topic.to_string(),
            readiness: Some((readiness, xsub_endpoints.to_vec())),
        })
    }
}

#[async_trait]
impl EventTransportTx for ZmqPubTransport {
    async fn wait_ready(&self) -> Result<()> {
        if let Some((readiness, endpoints)) = &self.readiness {
            for endpoint in endpoints {
                readiness.wait_connected(endpoint).await?;
            }
        }
        Ok(())
    }

    async fn publish(&self, _subject: &str, envelope_bytes: Bytes) -> Result<()> {
        let codec = MsgpackCodec;
        let (publisher_id, sequence) = codec.decode_envelope_identity(&envelope_bytes)?;

        let frame = Frame::new(envelope_bytes);
        let frames = vec![
            self.topic.as_bytes().to_vec(),
            publisher_id.to_be_bytes().to_vec(),
            sequence.to_be_bytes().to_vec(),
            frame.encode().to_vec(),
        ];

        self.socket
            .lock()
            .await
            .send(Multipart::from(frames))
            .await?;

        Ok(())
    }

    fn kind(&self) -> EventTransportKind {
        EventTransportKind::Zmq
    }
}

/// ZMQ SUB transport for subscribing to events.
///
/// Uses a background async reader to fan out frames to multiple local subscribers.
pub struct ZmqSubTransport {
    broadcast_tx: broadcast::Sender<Bytes>,
    socket_pump_handle: Arc<AbortOnDropHandle<()>>,
}

/// One validated multipart message from a direct ZMQ publisher.
pub struct ZmqWireMessage {
    pub publisher_id: u64,
    pub sequence: u64,
    pub payload: Bytes,
}

pub type ZmqWireStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<ZmqWireMessage>> + Send>>;

/// One dynamically managed ZMQ SUB socket connected to several publishers.
///
/// The caller must keep this value in one task. ZMQ SUB sockets are not thread-safe.
pub struct DynamicZmqSubSocket {
    socket: MonitoredZmqSocket<Subscribe>,
    expected_topic: Vec<u8>,
    readiness: Option<ZmqConnectionReadiness>,
}

impl DynamicZmqSubSocket {
    /// Start a dynamically managed socket and expose its connection handshakes.
    pub fn connect_with_readiness(
        endpoint: &str,
        topic: &str,
        rcvhwm: i32,
    ) -> Result<(Self, ZmqConnectionReadiness)> {
        let (socket, readiness) = connect_subscriber_with_readiness(endpoint, topic, rcvhwm)?;
        Ok((
            Self {
                socket,
                expected_topic: topic.as_bytes().to_vec(),
                readiness: Some(readiness.clone()),
            },
            readiness,
        ))
    }

    /// Connect a new SUB socket with an explicit receive high-water mark.
    pub fn connect_with_rcvhwm(endpoint: &str, topic: &str, rcvhwm: i32) -> Result<Self> {
        let socket = ZmqSubTransport::connect_socket_with_rcvhwm(endpoint, topic, rcvhwm)?;
        tracing::info!(endpoint, topic, rcvhwm, "Dynamic ZMQ SUB socket connected");
        Ok(Self {
            socket: MonitoredZmqSocket::unmonitored(socket),
            expected_topic: topic.as_bytes().to_vec(),
            readiness: None,
        })
    }

    /// Connect this SUB socket to one more publisher endpoint.
    pub fn add_endpoint(&mut self, endpoint: &str) -> Result<()> {
        self.socket.get_socket().connect(endpoint)?;
        if let Some(readiness) = &self.readiness {
            readiness.record_endpoint(self.socket.get_socket(), endpoint)?;
        }
        Ok(())
    }

    /// Stop receiving from one publisher endpoint.
    pub fn remove_endpoint(&mut self, endpoint: &str) -> Result<()> {
        self.socket.get_socket().disconnect(endpoint)?;
        Ok(())
    }

    /// Receive and decode the next multipart message.
    pub async fn next(&mut self) -> Option<Result<ZmqWireMessage>> {
        loop {
            let frames = match self.socket.next().await? {
                Ok(frames) => frames,
                Err(error) => return Some(Err(error.into())),
            };
            match decode_multipart(frames, &self.expected_topic) {
                Ok(message) => return Some(Ok(message)),
                Err(error) => {
                    tracing::warn!(%error, "Dropping malformed dynamic ZMQ message");
                }
            }
        }
    }
}

/// One event envelope whose ZMQ frames and envelope attribution agree.
#[derive(Debug)]
pub struct ValidatedEnvelope {
    pub publisher_id: u64,
    pub sequence: u64,
    pub published_at: u64,
    pub payload: Bytes,
}

/// Failure returned while reading a [`ValidatedZmqSource`].
#[derive(Debug, Error)]
pub enum ValidatedZmqSourceError {
    #[error("direct ZMQ receive failed: {0}")]
    Receive(#[source] anyhow::Error),
    #[error("direct ZMQ envelope decode failed: {0}")]
    EnvelopeDecode(#[source] anyhow::Error),
    #[error(
        "direct ZMQ identity mismatch: expected publisher {expected_publisher_id} topic {expected_topic}, frame publisher {frame_publisher_id} sequence {frame_sequence}, envelope publisher {envelope_publisher_id} sequence {envelope_sequence} topic {envelope_topic}"
    )]
    IdentityMismatch {
        expected_publisher_id: u64,
        expected_topic: String,
        frame_publisher_id: u64,
        frame_sequence: u64,
        envelope_publisher_id: u64,
        envelope_sequence: u64,
        envelope_topic: String,
    },
}

/// A direct ZMQ source that validates frame and envelope attribution once.
pub struct ValidatedZmqSource {
    stream: ZmqWireStream,
    expected_topic: String,
    expected_publisher_id: u64,
    codec: Codec,
}

impl ValidatedZmqSource {
    /// Connect after the subscribed socket has completed its transport handshake.
    pub async fn connect_ready(
        endpoint: &str,
        topic: &str,
        expected_publisher_id: u64,
        rcvhwm: i32,
    ) -> Result<Self> {
        Ok(Self {
            stream: ZmqSubTransport::connect_single_consumer_ready(endpoint, topic, rcvhwm).await?,
            expected_topic: topic.to_string(),
            expected_publisher_id,
            codec: Codec::default(),
        })
    }

    pub async fn connect_default(
        endpoint: &str,
        topic: &str,
        expected_publisher_id: u64,
    ) -> Result<Self> {
        Self::connect(endpoint, topic, expected_publisher_id, ZMQ_RCVHWM).await
    }

    pub async fn connect(
        endpoint: &str,
        topic: &str,
        expected_publisher_id: u64,
        rcvhwm: i32,
    ) -> Result<Self> {
        Ok(Self {
            stream: ZmqSubTransport::connect_single_consumer_with_rcvhwm(endpoint, topic, rcvhwm)
                .await?,
            expected_topic: topic.to_string(),
            expected_publisher_id,
            codec: Codec::default(),
        })
    }

    pub async fn next(
        &mut self,
    ) -> Option<std::result::Result<ValidatedEnvelope, ValidatedZmqSourceError>> {
        let message = match self.stream.next().await? {
            Ok(message) => message,
            Err(error) => return Some(Err(ValidatedZmqSourceError::Receive(error))),
        };
        let envelope = match self.codec.decode_envelope(&message.payload) {
            Ok(envelope) => envelope,
            Err(error) => {
                return Some(Err(ValidatedZmqSourceError::EnvelopeDecode(error)));
            }
        };

        if envelope.publisher_id != self.expected_publisher_id
            || envelope.publisher_id != message.publisher_id
            || envelope.sequence != message.sequence
            || envelope.topic != self.expected_topic
        {
            return Some(Err(ValidatedZmqSourceError::IdentityMismatch {
                expected_publisher_id: self.expected_publisher_id,
                expected_topic: self.expected_topic.clone(),
                frame_publisher_id: message.publisher_id,
                frame_sequence: message.sequence,
                envelope_publisher_id: envelope.publisher_id,
                envelope_sequence: envelope.sequence,
                envelope_topic: envelope.topic,
            }));
        }

        Some(Ok(ValidatedEnvelope {
            publisher_id: envelope.publisher_id,
            sequence: envelope.sequence,
            published_at: envelope.published_at,
            payload: envelope.payload,
        }))
    }
}

impl ZmqSubTransport {
    /// Connect one consumer after the subscribed socket's transport handshake.
    pub async fn connect_single_consumer_ready(
        endpoint: &str,
        topic: &str,
        rcvhwm: i32,
    ) -> Result<ZmqWireStream> {
        let (socket, readiness) = connect_subscriber_with_readiness(endpoint, topic, rcvhwm)?;
        readiness.wait_connected(endpoint).await?;
        Self::single_consumer_stream(socket, topic)
    }

    /// Connect a consumer after every broker transport handshake completes.
    pub async fn connect_single_consumer_multiple_ready(
        endpoints: &[String],
        topic: &str,
    ) -> Result<ZmqWireStream> {
        let (first, rest) = endpoints
            .split_first()
            .context("Cannot connect to zero endpoints")?;
        let (socket, readiness) = connect_subscriber_with_readiness(first, topic, ZMQ_RCVHWM)?;
        for endpoint in rest {
            socket.get_socket().connect(endpoint)?;
            readiness.record_endpoint(socket.get_socket(), endpoint)?;
        }
        for endpoint in endpoints {
            readiness.wait_connected(endpoint).await?;
        }
        Self::single_consumer_stream(socket, topic)
    }

    fn connect_socket(endpoint: &str, topic: &str) -> Result<Subscribe> {
        Self::connect_socket_with_rcvhwm(endpoint, topic, ZMQ_RCVHWM)
    }

    fn connect_socket_with_rcvhwm(endpoint: &str, topic: &str, rcvhwm: i32) -> Result<Subscribe> {
        anyhow::ensure!(rcvhwm > 0, "ZMQ receive HWM must be greater than zero");
        let ctx = shared_zmq_context()?;
        let socket = connect_tmq_socket(
            configure_subscribe_builder_with_hwm(subscribe(&ctx), rcvhwm),
            endpoint,
        )?;
        Ok(socket.subscribe(topic.as_bytes())?)
    }

    /// Create a new ZMQ subscriber by connecting to a single endpoint.
    pub async fn connect(endpoint: &str, topic: &str) -> Result<Self> {
        let socket = Self::connect_socket(endpoint, topic)?;

        tracing::info!(
            endpoint = %endpoint,
            topic = %topic,
            rcvhwm = ZMQ_RCVHWM,
            "ZMQ SUB transport connected with configured HWM"
        );

        let (broadcast_tx, _) = broadcast::channel(1024);
        let pump_handle = Self::start_socket_pump(socket, broadcast_tx.clone());

        Ok(Self {
            broadcast_tx,
            socket_pump_handle: Arc::new(AbortOnDropHandle::new(pump_handle)),
        })
    }

    /// Connect one consumer directly to one ZMQ publisher.
    ///
    /// Unlike [`Self::connect`], this stream owns and polls the socket directly. It
    /// therefore has no background pump or lossy broadcast hop and naturally
    /// applies backpressure at the configured ZMQ receive HWM.
    pub async fn connect_single_consumer(endpoint: &str, topic: &str) -> Result<ZmqWireStream> {
        Self::connect_single_consumer_with_rcvhwm(endpoint, topic, ZMQ_RCVHWM).await
    }

    /// Connect one consumer directly to one ZMQ publisher with an explicit receive HWM.
    pub async fn connect_single_consumer_with_rcvhwm(
        endpoint: &str,
        topic: &str,
        rcvhwm: i32,
    ) -> Result<ZmqWireStream> {
        let socket = Self::connect_socket_with_rcvhwm(endpoint, topic, rcvhwm)?;
        tracing::info!(
            endpoint,
            topic,
            rcvhwm,
            "ZMQ single-consumer stream connected"
        );
        Self::single_consumer_stream(socket, topic)
    }

    /// Connect one consumer directly to multiple ZMQ endpoints.
    ///
    /// The returned stream owns one SUB socket connected to every endpoint. It
    /// avoids the lossy local broadcast hop used by [`Self::connect_multiple`].
    pub async fn connect_single_consumer_multiple(
        endpoints: &[String],
        topic: &str,
    ) -> Result<ZmqWireStream> {
        let mut endpoint_iter = endpoints.iter();
        let Some(first_endpoint) = endpoint_iter.next() else {
            anyhow::bail!("Cannot connect to zero endpoints");
        };

        let endpoint_count = endpoints.len();
        let socket = Self::connect_socket(first_endpoint, topic)?;
        for endpoint in endpoint_iter {
            socket.get_socket().connect(endpoint)?;
        }

        tracing::info!(
            endpoint_count,
            topic,
            rcvhwm = ZMQ_RCVHWM,
            "ZMQ single-consumer stream connected to multiple endpoints"
        );
        tracing::debug!(?endpoints, topic, "ZMQ single-consumer stream endpoints");
        Self::single_consumer_stream(socket, topic)
    }

    fn single_consumer_stream<S>(mut socket: S, topic: &str) -> Result<ZmqWireStream>
    where
        S: Stream<Item = tmq::Result<Multipart>> + Send + Unpin + 'static,
    {
        let expected_topic = topic.as_bytes().to_vec();

        let stream = stream! {
            while let Some(result) = socket.next().await {
                let frames = match result {
                    Ok(frames) => frames,
                    Err(error) => {
                        yield Err(error.into());
                        break;
                    }
                };

                match decode_multipart(frames, &expected_topic) {
                    Ok(message) => yield Ok(message),
                    Err(error) => {
                        tracing::warn!(%error, "Dropping malformed ZMQ message");
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    /// Connect to broker's XPUB endpoint (broker mode)
    pub async fn connect_broker(xpub_endpoint: &str, topic: &str) -> Result<Self> {
        Self::connect(xpub_endpoint, topic).await
    }

    /// Connect to multiple broker XPUB endpoints (HA mode)
    pub async fn connect_broker_multiple(xpub_endpoints: &[String], topic: &str) -> Result<Self> {
        Self::connect_multiple(xpub_endpoints, topic).await
    }

    /// Create a new ZMQ subscriber by connecting to multiple endpoints (fan-in).
    pub async fn connect_multiple(endpoints: &[String], topic: &str) -> Result<Self> {
        let mut endpoints_iter = endpoints.iter();
        let Some(first_endpoint) = endpoints_iter.next() else {
            anyhow::bail!("Cannot connect to zero endpoints");
        };

        let ctx = shared_zmq_context()?;
        let socket =
            connect_tmq_socket(configure_subscribe_builder(subscribe(&ctx)), first_endpoint)?
                .subscribe(topic.as_bytes())?;

        for endpoint in endpoints_iter {
            socket.get_socket().connect(endpoint)?;
            tracing::debug!(endpoint = %endpoint, "ZMQ SUB connected to endpoint");
        }

        tracing::info!(
            num_endpoints = endpoints.len(),
            topic = %topic,
            rcvhwm = ZMQ_RCVHWM,
            "ZMQ SUB transport connected to multiple endpoints with configured HWM"
        );

        let (broadcast_tx, _) = broadcast::channel(1024);
        let pump_handle = Self::start_socket_pump(socket, broadcast_tx.clone());

        Ok(Self {
            broadcast_tx,
            socket_pump_handle: Arc::new(AbortOnDropHandle::new(pump_handle)),
        })
    }

    fn start_socket_pump(
        mut socket: Subscribe,
        broadcast_tx: broadcast::Sender<Bytes>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let Some(result) = socket.next().await else {
                    tracing::info!("ZMQ socket stream ended");
                    break;
                };

                let frames = match result {
                    Ok(frames) => frames,
                    Err(error) => {
                        tracing::error!(error = %error, "ZMQ receive error in socket pump");
                        break;
                    }
                };

                match decode_multipart(frames, &[]) {
                    Ok(message) => {
                        tracing::trace!(
                            publisher_id = message.publisher_id,
                            sequence = message.sequence,
                            "Socket pump received ZMQ message"
                        );
                        let _ = broadcast_tx.send(message.payload);
                    }
                    Err(error) => {
                        tracing::warn!(error = %error, "Failed to decode ZMQ frame in socket pump");
                    }
                }
            }

            tracing::info!("ZMQ socket pump task terminated");
        })
    }
}

fn decode_multipart(mut frames: Multipart, expected_topic: &[u8]) -> Result<ZmqWireMessage> {
    if frames.len() != 4 {
        anyhow::bail!("unexpected ZMQ multipart frame count: {}", frames.len());
    }

    if !expected_topic.is_empty() && &frames[0][..] != expected_topic {
        anyhow::bail!("ZMQ message topic disagrees with the exact subscription topic");
    }

    let publisher_id_bytes = &frames[1];
    if publisher_id_bytes.len() != 8 {
        anyhow::bail!(
            "invalid ZMQ publisher ID frame length: {}",
            publisher_id_bytes.len()
        );
    }
    let publisher_id = u64::from_be_bytes(publisher_id_bytes[..].try_into().unwrap());

    let sequence_bytes = &frames[2];
    if sequence_bytes.len() != 8 {
        anyhow::bail!(
            "invalid ZMQ sequence frame length: {}",
            sequence_bytes.len()
        );
    }
    let sequence = u64::from_be_bytes(sequence_bytes[..].try_into().unwrap());

    let frame_message = frames
        .pop_back()
        .ok_or_else(|| anyhow!("ZMQ multipart message has no payload frame"))?;
    let frame_bytes = Bytes::from_owner(ZmqMessageOwner(frame_message));
    let frame = Frame::decode(frame_bytes)?;

    Ok(ZmqWireMessage {
        publisher_id,
        sequence,
        payload: frame.payload,
    })
}

#[async_trait]
impl EventTransportRx for ZmqSubTransport {
    async fn subscribe(&self, _subject: &str) -> Result<WireStream> {
        let mut receiver = self.broadcast_tx.subscribe();
        let socket_pump_handle = Arc::clone(&self.socket_pump_handle);

        let stream = stream! {
            // Keep the socket pump alive after the transport is dropped. The
            // final transport or subscription stream aborts the pump, which
            // drops its owned ZMQ socket instead of detaching the task.
            let _socket_pump_handle = socket_pump_handle;
            loop {
                match receiver.recv().await {
                    Ok(payload) => yield Ok(payload),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped = skipped, "Subscriber lagged behind, skipped messages");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tracing::info!("Broadcast channel closed");
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn kind(&self) -> EventTransportKind {
        EventTransportKind::Zmq
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn configures_zmq_io_threads() {
        for (value, expected) in [(None, 4), (Some("1"), 1)] {
            let context = super::configured_zmq_context(value.map(OsStr::new)).unwrap();
            assert_eq!(context.get_io_threads().unwrap(), expected);
        }
        for value in ["0", "invalid", "2147483648"] {
            assert!(super::configured_zmq_context(Some(OsStr::new(value))).is_err());
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(super::configured_zmq_context(Some(OsStr::from_bytes(b"\xff"))).is_err());
        }
    }

    use super::*;
    use crate::transports::event_plane::{EventEnvelope, MsgpackCodec};
    use std::collections::HashSet;
    use tokio::time::{Duration, timeout};

    #[tokio::test]
    async fn subscriber_readiness_waits_for_handshake_with_hostname() {
        // Reserve the port with a plain TCP listener: a TCP connection alone is
        // insufficient, and no ZMTP publisher exists until the explicit bind.
        let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = reservation.local_addr().unwrap().port();
        let endpoint = format!("tcp://localhost:{port}");
        let (_socket, readiness) =
            connect_subscriber_with_readiness(&endpoint, "kv-events", 128).unwrap();
        assert!(
            timeout(
                Duration::from_millis(25),
                readiness.wait_connected(&endpoint)
            )
            .await
            .is_err(),
            "socket construction or a plain TCP connection must not report ready"
        );

        drop(reservation);
        let (_publisher, _) =
            ZmqPubTransport::bind(&format!("tcp://127.0.0.1:{port}"), "kv-events")
                .await
                .unwrap();
        timeout(Duration::from_secs(5), readiness.wait_connected(&endpoint))
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "publisher handshake did not arrive: {error}; aliases={:?}, connected={:?}",
                    readiness.endpoints.lock(),
                    readiness.connected.borrow()
                )
            })
            .unwrap();
    }

    #[tokio::test]
    async fn subscriber_readiness_teardown_preserves_shared_context() {
        // Closing a monitored socket must not leave a libzmq I/O thread blocked
        // trying to send to a monitor receiver that has already been dropped.
        // Reuse the same context across enough connections to exercise every I/O
        // thread, and require the first publication after each connection.
        let context = shared_zmq_context().unwrap();
        let topic = "monitor-lifetime";
        for sequence in 0..16_u64 {
            let publisher = context.socket(zmq::XPUB).unwrap();
            publisher.set_linger(0).unwrap();
            publisher.set_rcvtimeo(2_000).unwrap();
            publisher.bind("tcp://127.0.0.1:*").unwrap();
            let endpoint = publisher.get_last_endpoint().unwrap().unwrap();
            let mut stream = timeout(Duration::from_secs(2), async {
                if sequence % 2 == 0 {
                    ZmqSubTransport::connect_single_consumer_ready(&endpoint, topic, 128).await
                } else {
                    ZmqSubTransport::connect_single_consumer_multiple_ready(&[endpoint], topic)
                        .await
                }
            })
            .await
            .expect("previous monitor teardown must not block another connection")
            .unwrap();
            let payload = encoded_event(topic, 23, sequence);
            let published = payload.clone();
            let send = tokio::task::spawn_blocking(move || {
                // XPUB observes the subscription so this teardown regression
                // does not depend on PUB/SUB's unrelated slow-joiner timing.
                let subscription = publisher.recv_bytes(0).unwrap();
                assert_eq!(&subscription[1..], topic.as_bytes());
                assert_eq!(subscription[0], 1);
                publisher
                    .send_multipart(
                        [
                            topic.as_bytes().to_vec(),
                            23_u64.to_be_bytes().to_vec(),
                            sequence.to_be_bytes().to_vec(),
                            Frame::new(published).encode().to_vec(),
                        ],
                        0,
                    )
                    .unwrap();
                publisher
            });
            let message = timeout(Duration::from_secs(2), stream.next())
                .await
                .expect("the first event must arrive after each connection")
                .unwrap()
                .unwrap();
            assert_eq!(message.publisher_id, 23);
            assert_eq!(message.sequence, sequence);
            assert_eq!(message.payload, payload);
            let publisher = send.await.unwrap();
            drop(stream);
            drop(publisher);
        }
    }

    #[test]
    fn emfile_errno_selects_source_specific_guidance() {
        assert_eq!(
            socket_limit_guidance(Some(libc::EMFILE), ZMQ_SOCKET_LIMIT_GUIDANCE),
            Some(ZMQ_SOCKET_LIMIT_GUIDANCE)
        );
        assert_eq!(
            socket_limit_guidance(Some(libc::EMFILE), PROCESS_FD_LIMIT_GUIDANCE),
            Some(PROCESS_FD_LIMIT_GUIDANCE)
        );
        assert_eq!(
            socket_limit_guidance(Some(libc::EINVAL), ZMQ_SOCKET_LIMIT_GUIDANCE),
            None
        );
    }

    #[test]
    fn tmq_io_emfile_preserves_error_and_adds_fd_guidance() {
        let error = tmq::TmqError::Io(std::io::Error::from_raw_os_error(libc::EMFILE));
        let original = error.to_string();
        let message = map_socket_creation_error(error).to_string();

        assert!(message.starts_with(&original));
        assert!(message.contains(PROCESS_FD_LIMIT_GUIDANCE));
        assert!(!message.contains("ZMQ_MAX_SOCKETS"));
    }

    #[test]
    fn non_emfile_error_is_unchanged() {
        let error = tmq::TmqError::Io(std::io::Error::from_raw_os_error(libc::EINVAL));
        let original = error.to_string();

        assert_eq!(map_socket_creation_error(error).to_string(), original);
    }

    async fn send_raw(publisher: &ZmqPubTransport, frames: Vec<Vec<u8>>) {
        publisher
            .socket
            .lock()
            .await
            .send(Multipart::from(frames))
            .await
            .unwrap();
    }

    fn encoded_event(topic: &str, publisher_id: u64, sequence: u64) -> Bytes {
        MsgpackCodec
            .encode_envelope(&EventEnvelope {
                publisher_id,
                sequence,
                published_at: sequence,
                topic: topic.to_string(),
                payload: Bytes::from_static(b"payload"),
            })
            .unwrap()
    }

    async fn receive_publishers(
        subscriber: &mut DynamicZmqSubSocket,
        topic: &str,
        sequence: u64,
        expected: usize,
        publications: &[(&ZmqPubTransport, &Bytes)],
    ) -> HashSet<u64> {
        timeout(Duration::from_secs(2), async {
            let mut seen = HashSet::new();
            while seen.len() < expected {
                for (publisher, payload) in publications {
                    publisher.publish(topic, (*payload).clone()).await.unwrap();
                }
                if let Ok(Some(Ok(message))) =
                    timeout(Duration::from_millis(25), subscriber.next()).await
                    && message.sequence == sequence
                {
                    seen.insert(message.publisher_id);
                }
            }
            seen
        })
        .await
        .expect("dynamic subscriber should receive expected publishers")
    }

    #[test]
    fn test_zmq_message_owner_survives_clones_slices_and_thread_transfer() {
        let small = b"inline zmq message";
        let small_bytes = Bytes::from_owner(ZmqMessageOwner(Message::from(&small[..])));
        let small_clone = small_bytes.clone();
        drop(small_bytes);

        let small_slice = small_clone.slice(7..10);
        drop(small_clone);
        assert_eq!(small_slice, Bytes::from_static(b"zmq"));

        let large = vec![0x5a; 64 * 1024];
        let large_message = Message::from(large);
        let large_ptr = large_message.as_ptr();
        let large_bytes = Bytes::from_owner(ZmqMessageOwner(large_message));
        assert_eq!(large_bytes.as_ptr(), large_ptr);

        let large_clone = large_bytes.clone();
        drop(large_bytes);
        let returned = std::thread::spawn(move || {
            assert_eq!(large_clone.len(), 64 * 1024);
            assert!(large_clone.iter().all(|byte| *byte == 0x5a));
            large_clone.slice(1024..2048)
        })
        .join()
        .unwrap();

        assert_eq!(returned.len(), 1024);
        assert!(returned.iter().all(|byte| *byte == 0x5a));
    }

    #[tokio::test]
    async fn test_zmq_pubsub_basic() {
        let topic = "test-topic";

        let (publisher, endpoint) = ZmqPubTransport::bind("tcp://127.0.0.1:0", topic)
            .await
            .expect("Failed to create publisher");

        tokio::time::sleep(Duration::from_millis(100)).await;

        let subscriber = ZmqSubTransport::connect(&endpoint, topic)
            .await
            .expect("Failed to create subscriber");

        let mut stream = subscriber
            .subscribe(topic)
            .await
            .expect("Failed to create subscription");

        // Broker-mode callers retain only the returned stream. It must keep the
        // socket pump alive after the transport itself leaves scope.
        drop(subscriber);

        tokio::time::sleep(Duration::from_millis(100)).await;

        let codec = MsgpackCodec;
        let envelope = EventEnvelope {
            publisher_id: 12345,
            sequence: 1,
            published_at: 1700000000000,
            topic: topic.to_string(),
            payload: Bytes::from("test payload"),
        };

        let envelope_bytes = codec.encode_envelope(&envelope).unwrap();
        publisher.publish(topic, envelope_bytes).await.unwrap();

        let result = timeout(Duration::from_secs(2), stream.next()).await;
        assert!(result.is_ok(), "Timeout waiting for message");

        let received_bytes = result.unwrap().unwrap().unwrap();
        let decoded = codec.decode_envelope(&received_bytes).unwrap();

        assert_eq!(decoded.publisher_id, 12345);
        assert_eq!(decoded.sequence, 1);
        assert_eq!(decoded.topic, topic);
    }

    #[tokio::test]
    async fn single_consumer_applies_explicit_receive_hwm() {
        let endpoint = format!("inproc://dynamo-zmq-explicit-hwm-{}", std::process::id());
        let topic = "explicit-hwm";
        let (_publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();

        let socket = ZmqSubTransport::connect_socket_with_rcvhwm(&endpoint, topic, 37).unwrap();
        assert_eq!(socket.get_socket().get_rcvhwm().unwrap(), 37);

        let default_socket = ZmqSubTransport::connect_socket(&endpoint, topic).unwrap();
        assert_eq!(
            default_socket.get_socket().get_rcvhwm().unwrap(),
            ZMQ_RCVHWM
        );
        assert!(
            ZmqSubTransport::connect_single_consumer_with_rcvhwm(&endpoint, topic, 0)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn validated_source_rejects_bad_envelopes_and_continues_zero_copy() {
        let codec = Codec::default();
        let topic = "validated-source";
        let publisher_id = 7;
        let invalid = ZmqWireMessage {
            publisher_id,
            sequence: 0,
            payload: Bytes::from_static(&[0xc1]),
        };
        let mismatched_payload = codec
            .encode_envelope_parts(8, 1, 11, topic, b"mismatch")
            .unwrap();
        let mismatch = ZmqWireMessage {
            publisher_id,
            sequence: 1,
            payload: mismatched_payload,
        };
        let encoded = codec
            .encode_envelope_parts(publisher_id, 2, 12, topic, b"payload")
            .unwrap();
        let encoded_start = encoded.as_ptr() as usize;
        let encoded_end = encoded_start + encoded.len();
        let valid = ZmqWireMessage {
            publisher_id,
            sequence: 2,
            payload: encoded,
        };
        let mut source = ValidatedZmqSource {
            stream: Box::pin(futures::stream::iter(vec![
                Ok(invalid),
                Ok(mismatch),
                Ok(valid),
            ])),
            expected_topic: topic.to_string(),
            expected_publisher_id: publisher_id,
            codec,
        };

        assert!(matches!(
            source.next().await.unwrap(),
            Err(ValidatedZmqSourceError::EnvelopeDecode(_))
        ));
        assert!(matches!(
            source.next().await.unwrap(),
            Err(ValidatedZmqSourceError::IdentityMismatch { .. })
        ));
        let envelope = source.next().await.unwrap().unwrap();
        assert_eq!(envelope.publisher_id, publisher_id);
        assert_eq!(envelope.sequence, 2);
        assert_eq!(envelope.published_at, 12);
        assert_eq!(envelope.payload, Bytes::from_static(b"payload"));
        let payload_ptr = envelope.payload.as_ptr() as usize;
        assert!((encoded_start..encoded_end).contains(&payload_ptr));
        assert!(source.next().await.is_none());
    }

    #[tokio::test]
    async fn single_consumer_preserves_wire_identity_and_exact_topic() {
        let endpoint = format!("inproc://dynamo-zmq-single-consumer-{}", std::process::id());
        let topic = "single-consumer";
        let (publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();
        let mut stream = ZmqSubTransport::connect_single_consumer(&endpoint, topic)
            .await
            .unwrap();
        let codec = MsgpackCodec;
        let anchor = EventEnvelope {
            publisher_id: 41,
            sequence: 1,
            published_at: 1,
            topic: topic.to_string(),
            payload: Bytes::from_static(b"anchor"),
        };
        let anchor_bytes = codec.encode_envelope(&anchor).unwrap();

        let wire = timeout(Duration::from_secs(2), async {
            loop {
                publisher
                    .publish(topic, anchor_bytes.clone())
                    .await
                    .unwrap();
                if let Ok(Some(Ok(message))) =
                    timeout(Duration::from_millis(25), stream.next()).await
                {
                    break message;
                }
            }
        })
        .await
        .expect("single-consumer socket should become ready");
        assert_eq!(wire.publisher_id, anchor.publisher_id);
        assert_eq!(wire.sequence, anchor.sequence);
        assert_eq!(
            codec.decode_envelope(&wire.payload).unwrap().payload,
            anchor.payload
        );

        let sentinel = EventEnvelope {
            publisher_id: 41,
            sequence: 2,
            published_at: 2,
            topic: topic.to_string(),
            payload: Bytes::from_static(b"sentinel"),
        };
        let sentinel_bytes = codec.encode_envelope(&sentinel).unwrap();
        let framed = Frame::new(sentinel_bytes.clone()).encode().to_vec();
        send_raw(
            &publisher,
            vec![
                format!("{topic}-prefix-collision").into_bytes(),
                sentinel.publisher_id.to_be_bytes().to_vec(),
                sentinel.sequence.to_be_bytes().to_vec(),
                framed,
            ],
        )
        .await;
        publisher.publish(topic, sentinel_bytes).await.unwrap();

        let wire = timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("valid event should follow an exact-topic rejection")
            .expect("single-consumer stream should remain open")
            .expect("valid event should decode");
        assert_eq!(wire.publisher_id, sentinel.publisher_id);
        assert_eq!(wire.sequence, sentinel.sequence);
        assert_eq!(
            codec.decode_envelope(&wire.payload).unwrap().payload,
            sentinel.payload
        );
    }

    #[tokio::test]
    async fn dynamic_sub_socket_adds_and_removes_publishers() {
        let process = std::process::id();
        let endpoint_a = format!("inproc://dynamo-zmq-dynamic-a-{process}");
        let endpoint_b = format!("inproc://dynamo-zmq-dynamic-b-{process}");
        let topic = "dynamic-subscriber";
        let (publisher_a, _) = ZmqPubTransport::bind(&endpoint_a, topic).await.unwrap();
        let (publisher_b, _) = ZmqPubTransport::bind(&endpoint_b, topic).await.unwrap();
        let mut subscriber =
            DynamicZmqSubSocket::connect_with_rcvhwm(&endpoint_a, topic, ZMQ_RCVHWM).unwrap();
        subscriber.add_endpoint(&endpoint_b).unwrap();

        let encoded_a = encoded_event(topic, 101, 1);
        let encoded_b = encoded_event(topic, 202, 1);
        let seen = receive_publishers(
            &mut subscriber,
            topic,
            1,
            2,
            &[(&publisher_a, &encoded_a), (&publisher_b, &encoded_b)],
        )
        .await;
        assert_eq!(seen, HashSet::from([101, 202]));

        subscriber.remove_endpoint(&endpoint_a).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let encoded_a_after_removal = encoded_event(topic, 101, 2);
        let encoded_b_after_removal = encoded_event(topic, 202, 2);
        let publications = [
            (&publisher_a, &encoded_a_after_removal),
            (&publisher_b, &encoded_b_after_removal),
        ];
        assert_eq!(
            receive_publishers(&mut subscriber, topic, 2, 1, &publications).await,
            HashSet::from([202])
        );

        let removed_delivery = timeout(Duration::from_millis(250), async {
            loop {
                publisher_a
                    .publish(topic, encoded_a_after_removal.clone())
                    .await
                    .unwrap();
                if let Some(Ok(message)) = subscriber.next().await
                    && message.publisher_id == 101
                    && message.sequence == 2
                {
                    return;
                }
            }
        })
        .await;
        assert!(
            removed_delivery.is_err(),
            "removed publisher delivered data"
        );
    }

    #[tokio::test]
    async fn test_zmq_socket_pump_stops_with_last_owner() {
        let endpoint = format!("inproc://dynamo-zmq-pump-lifetime-{}", std::process::id());
        let topic = "pump-lifetime";

        let (_publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();
        let subscriber = ZmqSubTransport::connect(&endpoint, topic).await.unwrap();
        let pump_handle = subscriber.socket_pump_handle.abort_handle();
        let stream = subscriber.subscribe(topic).await.unwrap();

        drop(subscriber);
        tokio::task::yield_now().await;
        assert!(
            !pump_handle.is_finished(),
            "subscription stream should keep the socket pump alive"
        );

        drop(stream);
        timeout(Duration::from_secs(1), async {
            while !pump_handle.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("socket pump should stop when its final owner is dropped");
    }

    #[tokio::test]
    async fn test_zmq_multiple_messages() {
        let port = 25556;
        let endpoint = format!("tcp://127.0.0.1:{port}");
        let topic = "multi-test";

        let (publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let subscriber = ZmqSubTransport::connect(&endpoint, topic).await.unwrap();
        let mut stream = subscriber.subscribe(topic).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let codec = MsgpackCodec;

        for i in 0..5 {
            let envelope = EventEnvelope {
                publisher_id: 99999,
                sequence: i,
                published_at: 1700000000000 + i,
                topic: topic.to_string(),
                payload: Bytes::from(format!("message {i}")),
            };

            let bytes = codec.encode_envelope(&envelope).unwrap();
            publisher.publish(topic, bytes).await.unwrap();
        }

        for i in 0..5 {
            let result = timeout(Duration::from_secs(2), stream.next()).await;
            assert!(result.is_ok(), "Timeout on message {i}");

            let received = result.unwrap().unwrap().unwrap();
            let decoded = codec.decode_envelope(&received).unwrap();
            assert_eq!(decoded.sequence, i);
            assert_eq!(decoded.topic, topic);
        }
    }

    #[tokio::test]
    async fn test_zmq_socket_pump_continues_after_malformed_messages() {
        let endpoint = format!("inproc://dynamo-zmq-malformed-{}", std::process::id());
        let topic = "malformed-test";

        let (publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();
        let subscriber = ZmqSubTransport::connect(&endpoint, topic).await.unwrap();
        let mut stream = subscriber.subscribe(topic).await.unwrap();

        let codec = MsgpackCodec;
        let anchor = EventEnvelope {
            publisher_id: 12345,
            sequence: 0,
            published_at: 1700000000000,
            topic: topic.to_string(),
            payload: Bytes::from_static(b"anchor"),
        };
        let anchor_bytes = codec.encode_envelope(&anchor).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let received_anchor = loop {
            publisher
                .publish(topic, anchor_bytes.clone())
                .await
                .unwrap();
            if let Ok(Some(Ok(bytes))) = timeout(Duration::from_millis(25), stream.next()).await {
                break bytes;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "timeout waiting for subscriber readiness anchor"
            );
        };
        assert_eq!(
            codec.decode_envelope(&received_anchor).unwrap().payload,
            anchor.payload
        );

        let topic_frame = topic.as_bytes().to_vec();
        let publisher_frame = 12345_u64.to_be_bytes().to_vec();
        let sequence_frame = 1_u64.to_be_bytes().to_vec();
        let empty_frame = Frame::new(Bytes::new()).encode().to_vec();

        send_raw(
            &publisher,
            vec![
                topic_frame.clone(),
                publisher_frame.clone(),
                sequence_frame.clone(),
            ],
        )
        .await;
        send_raw(
            &publisher,
            vec![
                topic_frame.clone(),
                publisher_frame.clone(),
                sequence_frame.clone(),
                empty_frame.clone(),
                b"extra".to_vec(),
            ],
        )
        .await;
        send_raw(
            &publisher,
            vec![
                topic_frame.clone(),
                vec![0; 7],
                sequence_frame.clone(),
                empty_frame.clone(),
            ],
        )
        .await;
        send_raw(
            &publisher,
            vec![
                topic_frame.clone(),
                publisher_frame.clone(),
                vec![0; 7],
                empty_frame,
            ],
        )
        .await;
        send_raw(
            &publisher,
            vec![
                topic_frame,
                publisher_frame,
                sequence_frame,
                vec![99, 0, 0, 0, 0],
            ],
        )
        .await;

        let sentinel = EventEnvelope {
            publisher_id: 12345,
            sequence: 2,
            published_at: 1700000000000,
            topic: topic.to_string(),
            payload: Bytes::from_static(b"sentinel"),
        };
        publisher
            .publish(topic, codec.encode_envelope(&sentinel).unwrap())
            .await
            .unwrap();

        let decoded = timeout(Duration::from_secs(2), async {
            loop {
                let received = stream.next().await.unwrap().unwrap();
                let decoded = codec.decode_envelope(&received).unwrap();
                if decoded.sequence == sentinel.sequence {
                    break decoded;
                }
            }
        })
        .await
        .expect("timeout waiting for valid message after malformed messages");
        assert_eq!(decoded.publisher_id, sentinel.publisher_id);
        assert_eq!(decoded.sequence, sentinel.sequence);
        assert_eq!(decoded.payload, sentinel.payload);
    }
}
