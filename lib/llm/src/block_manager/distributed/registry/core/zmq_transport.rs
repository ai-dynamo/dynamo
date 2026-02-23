// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ transport implementation.
//!
//! Uses DEALER/ROUTER for queries and PUSH/PULL for registrations.
//!
//! Each query is prefixed with a 4-byte monotonic request ID. The hub echoes
//! the ID back in the response. A dedicated socket task multiplexes all
//! concurrent callers over a single DEALER socket, routing responses back
//! via oneshot channels keyed by request ID.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tmq::{Context, Message, Multipart, dealer, push};
use tokio::sync::{mpsc, oneshot};

use super::transport::RegistryTransport;

/// Size of the request-ID prefix prepended to every query and response.
pub const REQUEST_ID_SIZE: usize = 4;

/// Default high-water mark for ZMQ sockets.
///
/// This limits the number of messages that can be queued before
/// sends start blocking or dropping messages.
pub const DEFAULT_HWM: i32 = 10_000;

/// Channel buffer size for the DEALER socket task.
const DEALER_CHANNEL_SIZE: usize = 256;

/// Channel buffer size for the PUSH socket task.
const PUSH_CHANNEL_SIZE: usize = 256;

/// Command sent from callers to the DEALER socket task.
struct DealerCommand {
    request_id: u32,
    /// Raw payload WITHOUT request ID prefix.
    payload: Vec<u8>,
    reply: oneshot::Sender<Result<Vec<u8>>>,
}

/// Command sent from callers to the PUSH socket task.
struct PushCommand {
    payload: Vec<u8>,
    reply: Option<oneshot::Sender<Result<()>>>,
}

/// Configuration for ZMQ transport.
#[derive(Clone, Debug)]
pub struct ZmqTransportConfig {
    pub query_addr: String,
    pub publish_addr: String,
    /// High-water mark for the DEALER (query) socket.
    /// Limits queued outbound queries.
    pub dealer_hwm: i32,
    /// High-water mark for the PUSH (publish) socket.
    /// Limits queued registrations - prevents unbounded memory growth.
    pub push_hwm: i32,
}

impl ZmqTransportConfig {
    pub fn new(query_addr: impl Into<String>, publish_addr: impl Into<String>) -> Self {
        Self {
            query_addr: query_addr.into(),
            publish_addr: publish_addr.into(),
            dealer_hwm: DEFAULT_HWM,
            push_hwm: DEFAULT_HWM,
        }
    }

    /// Set high-water marks for both sockets.
    ///
    /// Lower values reduce memory usage but may drop messages under load.
    /// Higher values buffer more but use more memory.
    pub fn with_hwm(mut self, hwm: i32) -> Self {
        self.dealer_hwm = hwm;
        self.push_hwm = hwm;
        self
    }

    /// Set high-water mark for the DEALER (query) socket.
    pub fn with_dealer_hwm(mut self, hwm: i32) -> Self {
        self.dealer_hwm = hwm;
        self
    }

    /// Set high-water mark for the PUSH (registration) socket.
    pub fn with_push_hwm(mut self, hwm: i32) -> Self {
        self.push_hwm = hwm;
        self
    }
}

impl Default for ZmqTransportConfig {
    fn default() -> Self {
        Self {
            query_addr: "tcp://localhost:5555".to_string(),
            publish_addr: "tcp://localhost:5556".to_string(),
            dealer_hwm: DEFAULT_HWM,
            push_hwm: DEFAULT_HWM,
        }
    }
}

/// ZMQ-based transport for distributed registry.
///
/// Uses two sockets, each owned by a dedicated tokio task:
/// - DEALER for request/response queries (with request-ID correlation)
/// - PUSH for fire-and-forget registrations
///
/// Callers submit requests via mpsc channels and receive responses via
/// oneshot channels. This allows concurrent callers to pipeline requests
/// over a single DEALER socket without mutex serialization.
///
/// Both sockets have configurable high-water marks to prevent
/// unbounded memory growth under load.
pub struct ZmqTransport {
    config: ZmqTransportConfig,
    dealer_tx: mpsc::Sender<DealerCommand>,
    push_tx: mpsc::Sender<PushCommand>,
    /// Monotonically increasing request ID for correlating responses.
    next_request_id: AtomicU32,
}

impl ZmqTransport {
    /// Connect to a registry hub.
    pub fn connect(config: ZmqTransportConfig) -> Result<Self> {
        // CRITICAL: Use separate ZMQ contexts for DEALER and PUSH sockets.
        // Each ZMQ context has 1 I/O thread by default. If both sockets share
        // a context, the PUSH socket's registration traffic (thousands of entries
        // during KVBM startup) starves the DEALER's ZMTP handshake and query
        // delivery, causing 30+ second delays on the first query.
        let dealer_context = Context::new();
        let push_context = Context::new();

        // Create DEALER socket with its own I/O thread
        let dealer = dealer::dealer(&dealer_context)
            .set_sndhwm(config.dealer_hwm)
            .set_rcvhwm(config.dealer_hwm)
            .connect(&config.query_addr)
            .map_err(|e| anyhow!("Failed to connect DEALER to {}: {}", config.query_addr, e))?;

        // Create PUSH socket with its own I/O thread
        let pusher = push::push(&push_context)
            .set_sndhwm(config.push_hwm)
            .connect(&config.publish_addr)
            .map_err(|e| anyhow!("Failed to connect PUSH to {}: {}", config.publish_addr, e))?;

        tracing::debug!(
            query_addr = %config.query_addr,
            push_addr = %config.publish_addr,
            dealer_hwm = config.dealer_hwm,
            push_hwm = config.push_hwm,
            "ZMQ transport connected"
        );

        // Create channels for the socket tasks
        let (dealer_tx, dealer_rx) = mpsc::channel::<DealerCommand>(DEALER_CHANNEL_SIZE);
        let (push_tx, push_rx) = mpsc::channel::<PushCommand>(PUSH_CHANNEL_SIZE);

        // Spawn dedicated socket tasks
        tokio::spawn(Self::dealer_task(dealer, dealer_rx));
        tokio::spawn(Self::push_task(pusher, push_rx));

        Ok(Self {
            config,
            dealer_tx,
            push_tx,
            next_request_id: AtomicU32::new(1),
        })
    }

    /// Connect with default configuration.
    pub fn connect_default() -> Result<Self> {
        Self::connect(ZmqTransportConfig::default())
    }

    /// Connect to specific host and ports.
    pub fn connect_to(host: &str, query_port: u16, publish_port: u16) -> Result<Self> {
        let config = ZmqTransportConfig::new(
            format!("tcp://{}:{}", host, query_port),
            format!("tcp://{}:{}", host, publish_port),
        );
        Self::connect(config)
    }

    /// Connect to specific host and ports with custom HWM.
    pub fn connect_to_with_hwm(
        host: &str,
        query_port: u16,
        publish_port: u16,
        hwm: i32,
    ) -> Result<Self> {
        let config = ZmqTransportConfig::new(
            format!("tcp://{}:{}", host, query_port),
            format!("tcp://{}:{}", host, publish_port),
        )
        .with_hwm(hwm);
        Self::connect(config)
    }

    /// Dedicated task that owns the DEALER socket.
    ///
    /// Multiplexes concurrent callers over a single socket:
    /// - Receives commands from mpsc channel, sends on DEALER
    /// - Receives responses from DEALER, routes to callers via oneshot
    /// - Stale responses (caller timed out) are discarded silently
    async fn dealer_task(
        mut dealer: dealer::Dealer,
        mut rx: mpsc::Receiver<DealerCommand>,
    ) {
        let mut in_flight: HashMap<u32, oneshot::Sender<Result<Vec<u8>>>> = HashMap::new();

        loop {
            tokio::select! {
                // New request from a caller
                cmd = rx.recv() => {
                    let Some(cmd) = cmd else {
                        // All senders dropped — transport is being dropped
                        tracing::debug!("DEALER task: channel closed, shutting down");
                        break;
                    };

                    // Prepend request ID to payload
                    let mut payload = Vec::with_capacity(REQUEST_ID_SIZE + cmd.payload.len());
                    payload.extend_from_slice(&cmd.request_id.to_le_bytes());
                    payload.extend_from_slice(&cmd.payload);

                    let mut msg = VecDeque::new();
                    msg.push_back(Message::from(payload));

                    if let Err(e) = dealer.send(Multipart(msg)).await {
                        // Send failed — notify caller
                        let _ = cmd.reply.send(Err(anyhow!("Failed to send request: {}", e)));
                        continue;
                    }

                    // Track this request for response routing
                    in_flight.insert(cmd.request_id, cmd.reply);
                }

                // Response from the hub
                result = dealer.next() => {
                    match result {
                        Some(Ok(msg)) => {
                            let frames: Vec<_> = msg.iter().collect();
                            if frames.is_empty() {
                                tracing::warn!("DEALER task: empty response");
                                continue;
                            }

                            let resp_bytes = &frames[0][..];
                            if resp_bytes.len() < REQUEST_ID_SIZE {
                                tracing::warn!(
                                    len = resp_bytes.len(),
                                    "DEALER task: response too short for request ID"
                                );
                                continue;
                            }

                            let resp_id = u32::from_le_bytes(
                                resp_bytes[..REQUEST_ID_SIZE].try_into().unwrap(),
                            );
                            let payload = resp_bytes[REQUEST_ID_SIZE..].to_vec();

                            if let Some(reply_tx) = in_flight.remove(&resp_id) {
                                // Route response to caller. If send fails, the caller
                                // already timed out / dropped the receiver — discard.
                                let _ = reply_tx.send(Ok(payload));
                            } else {
                                // No in-flight entry. This shouldn't happen since we always
                                // insert before sending, but could if a response arrives
                                // after the in_flight map was drained on shutdown.
                                tracing::warn!(
                                    resp_id,
                                    "DEALER task: response for unknown request ID"
                                );
                            }
                        }
                        Some(Err(e)) => {
                            tracing::error!(error = %e, "DEALER task: receive error");
                            // Drain all in-flight requests with errors
                            for (_, reply_tx) in in_flight.drain() {
                                let _ = reply_tx.send(Err(anyhow!("Socket error: {}", e)));
                            }
                            break;
                        }
                        None => {
                            tracing::warn!("DEALER task: socket closed");
                            for (_, reply_tx) in in_flight.drain() {
                                let _ = reply_tx.send(Err(anyhow!("Socket closed")));
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Drain any remaining in-flight requests
        for (_, reply_tx) in in_flight.drain() {
            let _ = reply_tx.send(Err(anyhow!("DEALER task shutting down")));
        }
    }

    /// Dedicated task that owns the PUSH socket.
    ///
    /// Simple fire-and-forget loop: receives commands, sends on PUSH socket.
    async fn push_task(
        mut pusher: push::Push,
        mut rx: mpsc::Receiver<PushCommand>,
    ) {
        while let Some(cmd) = rx.recv().await {
            let mut msg = VecDeque::new();
            msg.push_back(Message::from(cmd.payload));

            let result = pusher
                .send(Multipart(msg))
                .await
                .map_err(|e| anyhow!("Failed to push: {}", e));

            if let Some(reply) = cmd.reply {
                let _ = reply.send(result);
            }
        }

        tracing::debug!("PUSH task: channel closed, shutting down");
    }
}

#[async_trait]
impl RegistryTransport for ZmqTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        let t0 = std::time::Instant::now();

        // Generate a monotonic request ID
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        // Create oneshot for response routing
        let (reply_tx, reply_rx) = oneshot::channel();

        // Submit to the DEALER socket task
        self.dealer_tx
            .send(DealerCommand {
                request_id,
                payload: data.to_vec(),
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow!("DEALER socket task has shut down"))?;

        tracing::info!(
            request_id,
            data_len = data.len(),
            "ZMQ request submitted"
        );

        // Wait for response from the socket task.
        // There is no transport-level timeout here. The caller (RemoteHandle)
        // wraps every operation in REGISTRY_TIMEOUT. When that fires, this
        // future is dropped, the oneshot Receiver is dropped, and the socket
        // task will discard the late response when it arrives.
        let result = reply_rx
            .await
            .map_err(|_| anyhow!("DEALER socket task dropped response channel"))?;

        let total_ms = t0.elapsed().as_millis();
        match &result {
            Ok(payload) => {
                tracing::info!(
                    request_id,
                    total_ms,
                    resp_len = payload.len(),
                    "ZMQ response received"
                );
            }
            Err(e) => {
                tracing::warn!(
                    request_id,
                    total_ms,
                    error = %e,
                    "ZMQ request failed"
                );
            }
        }

        result
    }

    async fn publish(&self, data: &[u8]) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.push_tx
            .send(PushCommand {
                payload: data.to_vec(),
                reply: Some(reply_tx),
            })
            .await
            .map_err(|_| anyhow!("PUSH socket task has shut down"))?;

        reply_rx
            .await
            .map_err(|_| anyhow!("PUSH socket task dropped response channel"))?
    }

    fn name(&self) -> &'static str {
        "zmq"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = ZmqTransportConfig::new("tcp://host:1234", "tcp://host:5678").with_hwm(5000);

        assert_eq!(config.query_addr, "tcp://host:1234");
        assert_eq!(config.publish_addr, "tcp://host:5678");
        assert_eq!(config.dealer_hwm, 5000);
        assert_eq!(config.push_hwm, 5000);
    }

    #[test]
    fn test_separate_hwm() {
        let config = ZmqTransportConfig::new("tcp://host:1234", "tcp://host:5678")
            .with_dealer_hwm(1000)
            .with_push_hwm(2000);

        assert_eq!(config.dealer_hwm, 1000);
        assert_eq!(config.push_hwm, 2000);
    }

    #[test]
    fn test_default_hwm() {
        let config = ZmqTransportConfig::default();
        assert_eq!(config.dealer_hwm, DEFAULT_HWM);
        assert_eq!(config.push_hwm, DEFAULT_HWM);
    }
}
