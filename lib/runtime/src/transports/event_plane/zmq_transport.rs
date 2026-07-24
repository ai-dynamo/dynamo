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

use anyhow::{Result, anyhow};
use async_stream::stream;
use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tmq::{
    AsZmqSocket, Context, Message, Multipart, SocketBuilder,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::{Mutex, broadcast};
use tokio_util::task::AbortOnDropHandle;

/// Returns the process-wide shared ZMQ context.
///
/// libzmq spawns background I/O threads per `Context`, so all PUB/SUB sockets
/// share one. `zmq::Context` is reference-counted; clones drive the same context.
fn shared_zmq_context() -> Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(Context::new).clone()
}

/// High Water Mark (HWM) for ZMQ sockets.
/// This controls the maximum number of messages that can be queued.
/// Default ZMQ HWM is 1000, which limits scalability.
const ZMQ_SNDHWM: i32 = 100_000; // Send buffer: 100K messages
const ZMQ_RCVHWM: i32 = 100_000; // Receive buffer: 100K messages
const ZMQ_SNDTIMEOUT_MS: i32 = 0; // Send timeout: fail fast under pressure
const ZMQ_RCVTIMEOUT_MS: i32 = 100; // Receive timeout: 100ms (avoids blocking forever)

use super::codec::{Codec, MsgpackCodec};
use super::frame::Frame;
use super::transport::{EventTransportRx, EventTransportTx, WireStream};
use crate::discovery::EventTransportKind;

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
    socket: Arc<Mutex<Publish>>,
    topic: String,
}

impl ZmqPubTransport {
    /// Create a new ZMQ publisher by binding to an endpoint.
    ///
    /// If port is 0, finds an available port using TcpListener first,
    /// then binds ZMQ to that port.
    ///
    /// Returns the transport and the actual bound endpoint.
    pub async fn bind(endpoint: &str, topic: &str) -> Result<(Self, String)> {
        let actual_endpoint = if endpoint.ends_with(":0") {
            let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await?;
            let actual_addr = listener.local_addr()?;
            let port = actual_addr.port();
            drop(listener);

            format!("tcp://0.0.0.0:{port}")
        } else {
            endpoint.to_string()
        };

        let ctx = shared_zmq_context();
        let socket = configure_publish_builder(publish(&ctx)).bind(&actual_endpoint)?;

        tracing::info!(
            endpoint = %actual_endpoint,
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport bound with configured HWM"
        );

        Ok((
            Self {
                socket: Arc::new(Mutex::new(socket)),
                topic: topic.to_string(),
            },
            actual_endpoint,
        ))
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Connect to single broker XSUB endpoint (broker mode)
    pub async fn connect(xsub_endpoint: &str, topic: &str) -> Result<Self> {
        let ctx = shared_zmq_context();
        let socket = configure_publish_builder(publish(&ctx)).connect(xsub_endpoint)?;

        tracing::info!(
            endpoint = %xsub_endpoint,
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport connected to broker XSUB"
        );

        Ok(Self {
            socket: Arc::new(Mutex::new(socket)),
            topic: topic.to_string(),
        })
    }

    /// Connect to multiple broker XSUB endpoints (HA mode)
    pub async fn connect_multiple(xsub_endpoints: &[String], topic: &str) -> Result<Self> {
        let mut endpoints = xsub_endpoints.iter();
        let Some(first_endpoint) = endpoints.next() else {
            anyhow::bail!("Cannot connect to zero endpoints");
        };

        let ctx = shared_zmq_context();
        let socket = configure_publish_builder(publish(&ctx)).connect(first_endpoint)?;

        for endpoint in endpoints {
            socket.get_socket().connect(endpoint)?;
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
        })
    }
}

#[async_trait]
impl EventTransportTx for ZmqPubTransport {
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
    fn connect_socket(endpoint: &str, topic: &str) -> Result<Subscribe> {
        Self::connect_socket_with_rcvhwm(endpoint, topic, ZMQ_RCVHWM)
    }

    fn connect_socket_with_rcvhwm(endpoint: &str, topic: &str, rcvhwm: i32) -> Result<Subscribe> {
        anyhow::ensure!(rcvhwm > 0, "ZMQ receive HWM must be greater than zero");
        let ctx = shared_zmq_context();
        Ok(
            configure_subscribe_builder_with_hwm(subscribe(&ctx), rcvhwm)
                .connect(endpoint)?
                .subscribe(topic.as_bytes())?,
        )
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
        let mut socket = Self::connect_socket_with_rcvhwm(endpoint, topic, rcvhwm)?;
        let expected_topic = topic.as_bytes().to_vec();

        tracing::info!(
            endpoint,
            topic,
            rcvhwm,
            "Direct ZMQ single-consumer stream connected"
        );

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
                    Ok(message) => yield Ok(ZmqWireMessage {
                        publisher_id: message.publisher_id,
                        sequence: message.sequence,
                        payload: message.payload,
                    }),
                    Err(error) => {
                        tracing::warn!(%error, "Dropping malformed direct-ZMQ message");
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

        let ctx = shared_zmq_context();
        let socket = configure_subscribe_builder(subscribe(&ctx))
            .connect(first_endpoint)?
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

struct DecodedZmqMessage {
    publisher_id: u64,
    sequence: u64,
    payload: Bytes,
}

fn decode_multipart(mut frames: Multipart, expected_topic: &[u8]) -> Result<DecodedZmqMessage> {
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

    Ok(DecodedZmqMessage {
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
    use super::*;
    use crate::transports::event_plane::{EventEnvelope, MsgpackCodec};
    use tokio::time::{Duration, timeout};

    async fn send_raw(publisher: &ZmqPubTransport, frames: Vec<Vec<u8>>) {
        publisher
            .socket
            .lock()
            .await
            .send(Multipart::from(frames))
            .await
            .unwrap();
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
        let port = 25555;
        let endpoint = format!("tcp://127.0.0.1:{port}");
        let topic = "test-topic";

        let (publisher, _actual_endpoint) = ZmqPubTransport::bind(&endpoint, topic)
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
