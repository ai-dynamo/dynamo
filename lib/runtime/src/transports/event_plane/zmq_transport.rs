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
use std::sync::Arc;
use tmq::{
    AsZmqSocket, Context, Multipart, SocketBuilder,
    publish::{Publish, publish},
    subscribe::{Subscribe, subscribe},
};
use tokio::sync::{Mutex, broadcast};

/// High Water Mark (HWM) for ZMQ sockets.
/// This controls the maximum number of messages that can be queued.
/// Default ZMQ HWM is 1000, which limits scalability.
const ZMQ_SNDHWM: i32 = 100_000; // Send buffer: 100K messages
const ZMQ_RCVHWM: i32 = 100_000; // Receive buffer: 100K messages
const ZMQ_SNDTIMEOUT_MS: i32 = 0; // Send timeout: fail fast under pressure
const ZMQ_RCVTIMEOUT_MS: i32 = 100; // Receive timeout: 100ms (avoids blocking forever)

use super::codec::MsgpackCodec;
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
    builder
        .set_rcvhwm(ZMQ_RCVHWM)
        .set_rcvtimeo(ZMQ_RCVTIMEOUT_MS)
}

fn multipart_message(multipart: Multipart) -> Vec<Vec<u8>> {
    multipart.into_iter().map(|frame| frame.to_vec()).collect()
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

        let ctx = Context::new();
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
        let ctx = Context::new();
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

        let ctx = Context::new();
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
        let envelope = codec.decode_envelope(&envelope_bytes)?;

        let frame = Frame::new(envelope_bytes);
        let frames = vec![
            self.topic.as_bytes().to_vec(),
            envelope.publisher_id.to_be_bytes().to_vec(),
            envelope.sequence.to_be_bytes().to_vec(),
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
    _socket_pump_handle: tokio::task::JoinHandle<()>,
}

impl ZmqSubTransport {
    /// Create a new ZMQ subscriber by connecting to a single endpoint.
    pub async fn connect(endpoint: &str, topic: &str) -> Result<Self> {
        let ctx = Context::new();
        let socket = configure_subscribe_builder(subscribe(&ctx))
            .connect(endpoint)?
            .subscribe(topic.as_bytes())?;

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
            _socket_pump_handle: pump_handle,
        })
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

        let ctx = Context::new();
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
            _socket_pump_handle: pump_handle,
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
                    Ok(frames) => multipart_message(frames),
                    Err(error) => {
                        tracing::error!(error = %error, "ZMQ receive error in socket pump");
                        break;
                    }
                };

                if frames.len() != 4 {
                    tracing::warn!(
                        frame_count = frames.len(),
                        "Unexpected multipart frame count in socket pump"
                    );
                    continue;
                }

                let publisher_id_bytes = &frames[1];
                if publisher_id_bytes.len() != 8 {
                    tracing::warn!(
                        actual = publisher_id_bytes.len(),
                        "Invalid publisher_id frame in socket pump"
                    );
                    continue;
                }
                let publisher_id =
                    u64::from_be_bytes(publisher_id_bytes.as_slice().try_into().unwrap());

                let sequence_bytes = &frames[2];
                if sequence_bytes.len() != 8 {
                    tracing::warn!(
                        actual = sequence_bytes.len(),
                        "Invalid sequence frame in socket pump"
                    );
                    continue;
                }
                let sequence = u64::from_be_bytes(sequence_bytes.as_slice().try_into().unwrap());

                tracing::trace!(
                    publisher_id = publisher_id,
                    sequence = sequence,
                    "Socket pump received ZMQ message"
                );

                let frame_bytes = Bytes::from(frames[3].clone());
                match Frame::decode(frame_bytes) {
                    Ok(frame) => {
                        let _ = broadcast_tx.send(frame.payload);
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

#[async_trait]
impl EventTransportRx for ZmqSubTransport {
    async fn subscribe(&self, _subject: &str) -> Result<WireStream> {
        let mut receiver = self.broadcast_tx.subscribe();

        let stream = stream! {
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
}
