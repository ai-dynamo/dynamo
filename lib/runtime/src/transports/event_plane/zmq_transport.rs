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

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use bytes::Bytes;
use std::sync::{Arc, Mutex};

/// High Water Mark (HWM) for ZMQ sockets.
/// This controls the maximum number of messages that can be queued.
/// Default ZMQ HWM is 1000, which limits scalability.
const ZMQ_SNDHWM: i32 = 100_000; // Send buffer: 100K messages
const ZMQ_RCVHWM: i32 = 100_000; // Receive buffer: 100K messages

use super::codec::MsgpackCodec;
use super::frame::Frame;
use super::transport::{EventTransportRx, EventTransportTx, WireStream};
use crate::discovery::EventTransportKind;

/// ZMQ PUB transport for publishing events.
///
/// Uses raw zmq::Socket with configured HWM for better scalability.
pub struct ZmqPubTransport {
    socket: Arc<Mutex<zmq::Socket>>,
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
        // Parse the endpoint to check if we need to find an available port
        let actual_endpoint = if endpoint.ends_with(":0") {
            // Find an available port using TcpListener
            let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await?;
            let actual_addr = listener.local_addr()?;
            let port = actual_addr.port();
            drop(listener); // Close listener so ZMQ can bind to the port

            format!("tcp://0.0.0.0:{}", port)
        } else {
            endpoint.to_string()
        };

        // Create raw ZMQ socket with HWM configuration
        let endpoint_for_closure = actual_endpoint.clone();
        let socket = tokio::task::spawn_blocking(move || -> Result<zmq::Socket> {
            let ctx = zmq::Context::new();
            let socket = ctx.socket(zmq::PUB)?;

            // Configure High Water Mark for better scalability
            socket.set_sndhwm(ZMQ_SNDHWM)?;

            // Set send timeout to 0 (non-blocking)
            socket.set_sndtimeo(0)?;

            // Bind to endpoint
            socket.bind(&endpoint_for_closure)?;

            Ok(socket)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

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
        let endpoint_owned = xsub_endpoint.to_string();
        let topic_owned = topic.to_string();

        let socket = tokio::task::spawn_blocking(move || -> Result<zmq::Socket> {
            let ctx = zmq::Context::new();
            let socket = ctx.socket(zmq::PUB)?;

            // Configure High Water Mark for better scalability
            socket.set_sndhwm(ZMQ_SNDHWM)?;

            // Set send timeout to 0 (non-blocking)
            socket.set_sndtimeo(0)?;

            // Connect (not bind) to broker's XSUB
            socket.connect(&endpoint_owned)?;

            Ok(socket)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        tracing::info!(
            endpoint = %xsub_endpoint,
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport connected to broker XSUB"
        );

        Ok(Self {
            socket: Arc::new(Mutex::new(socket)),
            topic: topic_owned,
        })
    }

    /// Connect to multiple broker XSUB endpoints (HA mode)
    pub async fn connect_multiple(xsub_endpoints: &[String], topic: &str) -> Result<Self> {
        if xsub_endpoints.is_empty() {
            anyhow::bail!("Cannot connect to zero endpoints");
        }

        let endpoints_owned = xsub_endpoints.to_vec();
        let topic_owned = topic.to_string();

        let socket = tokio::task::spawn_blocking(move || -> Result<zmq::Socket> {
            let ctx = zmq::Context::new();
            let socket = ctx.socket(zmq::PUB)?;

            // Configure High Water Mark for better scalability
            socket.set_sndhwm(ZMQ_SNDHWM)?;

            // Set send timeout to 0 (non-blocking)
            socket.set_sndtimeo(0)?;

            // Connect to all XSUB endpoints (ZMQ handles load balancing)
            for endpoint in &endpoints_owned {
                socket.connect(endpoint)?;
                tracing::debug!(endpoint = %endpoint, "ZMQ PUB connected to broker XSUB");
            }

            Ok(socket)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        tracing::info!(
            num_endpoints = xsub_endpoints.len(),
            topic = %topic,
            sndhwm = ZMQ_SNDHWM,
            "ZMQ PUB transport connected to multiple broker XSUBs with configured HWM"
        );

        Ok(Self {
            socket: Arc::new(Mutex::new(socket)),
            topic: topic_owned,
        })
    }
}

#[async_trait]
impl EventTransportTx for ZmqPubTransport {
    async fn publish(&self, _subject: &str, envelope_bytes: Bytes) -> Result<()> {
        // Decode envelope to extract publisher_id and sequence for fast deduplication
        let codec = MsgpackCodec;
        let envelope = codec.decode_envelope(&envelope_bytes)?;

        // Create binary frame
        let frame = Frame::new(envelope_bytes);
        let frame_bytes = frame.encode();

        // Prepare multipart message: [topic, publisher_id, sequence, frame_bytes]
        let topic_bytes = self.topic.as_bytes().to_vec();
        let publisher_id_bytes = envelope.publisher_id.to_be_bytes().to_vec();
        let sequence_bytes = envelope.sequence.to_be_bytes().to_vec();
        let frame_vec = frame_bytes.to_vec();

        let socket = Arc::clone(&self.socket);
        tokio::task::spawn_blocking(move || -> Result<()> {
            let socket = socket.lock().unwrap();
            // Send topic frame (for ZMQ subscription filtering)
            socket.send(&topic_bytes, zmq::SNDMORE)?;
            // Send publisher_id (for fast deduplication)
            socket.send(&publisher_id_bytes, zmq::SNDMORE)?;
            // Send sequence (for fast deduplication)
            socket.send(&sequence_bytes, zmq::SNDMORE)?;
            // Send data frame (complete envelope)
            socket.send(&frame_vec, 0)?;
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    fn kind(&self) -> EventTransportKind {
        EventTransportKind::Zmq
    }
}

/// ZMQ SUB transport for subscribing to events.
///
/// Uses a background socket pump to avoid holding the socket lock across stream lifetimes.
/// Multiple subscribers can receive events concurrently via broadcast channel.
pub struct ZmqSubTransport {
    socket: Arc<Mutex<zmq::Socket>>,
    broadcast_tx: tokio::sync::broadcast::Sender<Bytes>,
    _socket_pump_handle: tokio::task::JoinHandle<()>,
}

impl ZmqSubTransport {
    /// Create a new ZMQ subscriber by connecting to a single endpoint.
    pub async fn connect(endpoint: &str, topic: &str) -> Result<Self> {
        let endpoint_owned = endpoint.to_string();
        let topic_owned = topic.to_string();

        let socket = tokio::task::spawn_blocking(move || -> Result<zmq::Socket> {
            let ctx = zmq::Context::new();
            let socket = ctx.socket(zmq::SUB)?;

            // Configure High Water Mark for better scalability
            socket.set_rcvhwm(ZMQ_RCVHWM)?;

            // Set receive timeout to -1 (blocking)
            socket.set_rcvtimeo(-1)?;

            // Connect to endpoint
            socket.connect(&endpoint_owned)?;

            // Subscribe to topic
            socket.set_subscribe(topic_owned.as_bytes())?;

            Ok(socket)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        tracing::info!(
            endpoint = %endpoint,
            topic = %topic,
            rcvhwm = ZMQ_RCVHWM,
            "ZMQ SUB transport connected with configured HWM"
        );

        let socket = Arc::new(Mutex::new(socket));

        // Create broadcast channel for multiple subscribers
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(1024);

        // Start background socket pump
        let pump_handle = Self::start_socket_pump(Arc::clone(&socket), broadcast_tx.clone());

        Ok(Self {
            socket,
            broadcast_tx,
            _socket_pump_handle: pump_handle,
        })
    }

    /// Connect to broker's XPUB endpoint (broker mode)
    pub async fn connect_broker(xpub_endpoint: &str, topic: &str) -> Result<Self> {
        Self::connect(xpub_endpoint, topic).await
    }

    /// Connect to multiple broker XPUB endpoints (HA mode)
    /// Reuses existing connect_multiple implementation
    pub async fn connect_broker_multiple(xpub_endpoints: &[String], topic: &str) -> Result<Self> {
        Self::connect_multiple(xpub_endpoints, topic).await
    }

    /// Create a new ZMQ subscriber by connecting to multiple endpoints (fan-in).
    pub async fn connect_multiple(endpoints: &[String], topic: &str) -> Result<Self> {
        if endpoints.is_empty() {
            anyhow::bail!("Cannot connect to zero endpoints");
        }

        let endpoints_owned = endpoints.to_vec();
        let topic_owned = topic.to_string();

        let socket = tokio::task::spawn_blocking(move || -> Result<zmq::Socket> {
            let ctx = zmq::Context::new();
            let socket = ctx.socket(zmq::SUB)?;

            // Configure High Water Mark for better scalability
            socket.set_rcvhwm(ZMQ_RCVHWM)?;

            // Set receive timeout to -1 (blocking)
            socket.set_rcvtimeo(-1)?;

            // Connect to all endpoints
            for endpoint in &endpoints_owned {
                socket.connect(endpoint)?;
                tracing::debug!(endpoint = %endpoint, "ZMQ SUB connected to endpoint");
            }

            // Subscribe to topic
            socket.set_subscribe(topic_owned.as_bytes())?;

            Ok(socket)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        tracing::info!(
            num_endpoints = endpoints.len(),
            topic = %topic,
            rcvhwm = ZMQ_RCVHWM,
            "ZMQ SUB transport connected to multiple endpoints with configured HWM"
        );

        let socket = Arc::new(Mutex::new(socket));

        // Create broadcast channel for multiple subscribers
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(1024);

        // Start background socket pump
        let pump_handle = Self::start_socket_pump(Arc::clone(&socket), broadcast_tx.clone());

        Ok(Self {
            socket,
            broadcast_tx,
            _socket_pump_handle: pump_handle,
        })
    }

    /// Background task that reads from socket and broadcasts to all subscribers.
    ///
    /// This task holds the socket lock only briefly during each recv operation,
    /// allowing multiple subscribers to receive concurrently via broadcast channel.
    fn start_socket_pump(
        socket: Arc<Mutex<zmq::Socket>>,
        broadcast_tx: tokio::sync::broadcast::Sender<Bytes>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                // Receive multipart message in blocking task: [topic, publisher_id, sequence, frame_bytes]
                let socket_clone = Arc::clone(&socket);
                let result =
                    tokio::task::spawn_blocking(move || -> Result<(Vec<u8>, u64, u64, Vec<u8>)> {
                        let socket = socket_clone.lock().unwrap();

                        // Receive topic frame
                        let topic = socket.recv_bytes(0)?;

                        // Receive publisher_id frame (8 bytes, u64 big-endian)
                        let publisher_id_bytes = socket.recv_bytes(0)?;
                        if publisher_id_bytes.len() != 8 {
                            anyhow::bail!(
                                "Invalid publisher_id frame: expected 8 bytes, got {}",
                                publisher_id_bytes.len()
                            );
                        }
                        let publisher_id =
                            u64::from_be_bytes(publisher_id_bytes.try_into().unwrap());

                        // Receive sequence frame (8 bytes, u64 big-endian)
                        let sequence_bytes = socket.recv_bytes(0)?;
                        if sequence_bytes.len() != 8 {
                            anyhow::bail!(
                                "Invalid sequence frame: expected 8 bytes, got {}",
                                sequence_bytes.len()
                            );
                        }
                        let sequence = u64::from_be_bytes(sequence_bytes.try_into().unwrap());

                        // Receive data frame
                        let data = socket.recv_bytes(0)?;

                        Ok((topic, publisher_id, sequence, data))
                    })
                    .await;

                match result {
                    Ok(Ok((_topic, publisher_id, sequence, frame_bytes))) => {
                        // Log dedup metadata for debugging
                        tracing::trace!(
                            publisher_id = publisher_id,
                            sequence = sequence,
                            "Socket pump received ZMQ message"
                        );

                        // Parse binary frame
                        let frame_bytes = Bytes::from(frame_bytes);
                        match Frame::decode(frame_bytes) {
                            Ok(frame) => {
                                // Broadcast payload to all subscribers
                                // Ignore send errors (no receivers or lagging receivers)
                                let _ = broadcast_tx.send(frame.payload);
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to decode ZMQ frame in socket pump");
                                continue;
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "ZMQ receive error in socket pump");
                        break;
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Task join error in socket pump");
                        break;
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
        // Subscribe to broadcast channel (does not hold socket lock)
        let mut receiver = self.broadcast_tx.subscribe();

        let stream = stream! {
            loop {
                match receiver.recv().await {
                    Ok(payload) => {
                        yield Ok(payload);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(
                            skipped = skipped,
                            "Subscriber lagged behind, skipped messages"
                        );
                        // Continue receiving, don't break the stream
                        continue;
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
        let endpoint = format!("tcp://127.0.0.1:{}", port);
        let topic = "test-topic";

        let (publisher, _actual_endpoint) = ZmqPubTransport::bind(&endpoint, topic)
            .await
            .expect("Failed to create publisher");

        tokio::time::sleep(Duration::from_millis(100)).await;

        let subscriber = ZmqSubTransport::connect(&endpoint, topic)
            .await
            .expect("Failed to create subscriber");

        use futures::StreamExt;
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
        let endpoint = format!("tcp://127.0.0.1:{}", port);
        let topic = "multi-test";

        let (publisher, _) = ZmqPubTransport::bind(&endpoint, topic).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let subscriber = ZmqSubTransport::connect(&endpoint, topic).await.unwrap();
        use futures::StreamExt;
        let mut stream = subscriber.subscribe(topic).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let codec = MsgpackCodec;

        for i in 0..5 {
            let envelope = EventEnvelope {
                publisher_id: 99999,
                sequence: i,
                published_at: 1700000000000 + i,
                topic: topic.to_string(),
                payload: Bytes::from(format!("message {}", i)),
            };

            let bytes = codec.encode_envelope(&envelope).unwrap();
            publisher.publish(topic, bytes).await.unwrap();
        }

        for i in 0..5 {
            let result = timeout(Duration::from_secs(2), stream.next()).await;
            assert!(result.is_ok(), "Timeout on message {}", i);

            let received = result.unwrap().unwrap().unwrap();
            let decoded = codec.decode_envelope(&received).unwrap();
            assert_eq!(decoded.sequence, i);
            assert_eq!(decoded.topic, topic);
        }
    }
}
