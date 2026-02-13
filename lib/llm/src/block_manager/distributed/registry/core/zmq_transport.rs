// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ transport implementation.
//!
//! Uses DEALER/ROUTER for queries and PUSH/PULL for registrations.
//!
//! Each query is prefixed with a 4-byte monotonic request ID. The hub echoes
//! the ID back in the response. The client loops on `recv`, discarding any
//! response whose ID does not match the current request, which eliminates
//! the desynchronization spiral caused by timed-out requests leaving stale
//! responses in the socket buffer.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tmq::{Context, Message, Multipart, dealer, push};
use tokio::sync::Mutex;

use super::transport::RegistryTransport;

/// Size of the request-ID prefix prepended to every query and response.
pub const REQUEST_ID_SIZE: usize = 4;

/// Default high-water mark for ZMQ sockets.
///
/// This limits the number of messages that can be queued before
/// sends start blocking or dropping messages.
pub const DEFAULT_HWM: i32 = 10_000;

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
/// Uses two sockets:
/// - DEALER for request/response queries (with request-ID correlation)
/// - PUSH for fire-and-forget registrations
///
/// Both sockets have configurable high-water marks to prevent
/// unbounded memory growth under load.
pub struct ZmqTransport {
    config: ZmqTransportConfig,
    dealer: Mutex<dealer::Dealer>,
    pusher: Mutex<push::Push>,
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

        Ok(Self {
            config,
            dealer: Mutex::new(dealer),
            pusher: Mutex::new(pusher),
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
}

#[async_trait]
impl RegistryTransport for ZmqTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        let t0 = std::time::Instant::now();
        let mut socket = self.dealer.lock().await;
        let lock_ms = t0.elapsed().as_millis();

        // Generate a monotonic request ID and prepend it to the payload.
        // The hub echoes this ID back in the response so we can correlate
        // responses to requests and discard stale ones from timed-out requests.
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut payload = Vec::with_capacity(REQUEST_ID_SIZE + data.len());
        payload.extend_from_slice(&request_id.to_le_bytes());
        payload.extend_from_slice(data);

        let mut msg = VecDeque::new();
        msg.push_back(Message::from(payload));

        let send_start = std::time::Instant::now();
        socket
            .send(Multipart(msg))
            .await
            .map_err(|e| anyhow!("Failed to send request: {}", e))?;
        let send_ms = send_start.elapsed().as_millis();

        tracing::info!(
            request_id,
            lock_ms,
            send_ms,
            data_len = data.len(),
            "ZMQ request sent"
        );

        // Read responses in a loop, discarding any whose request ID doesn't
        // match ours. This handles stale responses from previously timed-out
        // requests without needing a drain loop or fragile timing.
        //
        // There is no transport-level timeout here. The caller (RemoteHandle)
        // wraps every operation in REGISTRY_TIMEOUT. When that fires, this
        // future is dropped (releasing the socket lock). The late response
        // will be discarded by ID on the next request.
        loop {
            match socket.next().await {
                Some(Ok(msg)) => {
                    let frames: Vec<_> = msg.iter().collect();
                    if frames.is_empty() {
                        return Err(anyhow!("Empty response"));
                    }
                    let resp_bytes = &frames[0][..];
                    if resp_bytes.len() < REQUEST_ID_SIZE {
                        return Err(anyhow!(
                            "Response too short ({} bytes, need at least {})",
                            resp_bytes.len(),
                            REQUEST_ID_SIZE
                        ));
                    }

                    let resp_id =
                        u32::from_le_bytes(resp_bytes[..REQUEST_ID_SIZE].try_into().unwrap());

                    if resp_id == request_id {
                        let total_ms = t0.elapsed().as_millis();
                        tracing::info!(
                            request_id,
                            total_ms,
                            resp_len = resp_bytes.len() - REQUEST_ID_SIZE,
                            "ZMQ response received"
                        );
                        // This is our response — strip the request ID prefix.
                        return Ok(resp_bytes[REQUEST_ID_SIZE..].to_vec());
                    }

                    // Stale response from a previous request — discard and keep reading.
                    tracing::debug!(
                        expected = request_id,
                        got = resp_id,
                        "Discarding stale response (request ID mismatch)"
                    );
                }
                Some(Err(e)) => return Err(anyhow!("Receive error: {}", e)),
                None => return Err(anyhow!("Socket closed")),
            }
        }
    }

    async fn publish(&self, data: &[u8]) -> Result<()> {
        let mut socket = self.pusher.lock().await;

        let mut msg = VecDeque::new();
        msg.push_back(Message::from(data.to_vec()));
        socket
            .send(Multipart(msg))
            .await
            .map_err(|e| anyhow!("Failed to push: {}", e))?;

        Ok(())
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
