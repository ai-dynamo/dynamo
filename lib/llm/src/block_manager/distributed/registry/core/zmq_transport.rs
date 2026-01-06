// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ZMQ transport implementation.
//!
//! Uses DEALER/ROUTER for queries and PUSH/PULL for registrations.

use std::collections::VecDeque;
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tmq::{dealer, push, Context, Message, Multipart};
use tokio::sync::Mutex;

use super::transport::RegistryTransport;

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
    pub request_timeout: Duration,
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
            request_timeout: Duration::from_secs(5),
            dealer_hwm: DEFAULT_HWM,
            push_hwm: DEFAULT_HWM,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
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
            request_timeout: Duration::from_secs(5),
            dealer_hwm: DEFAULT_HWM,
            push_hwm: DEFAULT_HWM,
        }
    }
}

/// ZMQ-based transport for distributed registry.
///
/// Uses two sockets:
/// - DEALER for request/response queries
/// - PUSH for fire-and-forget registrations
///
/// Both sockets have configurable high-water marks to prevent
/// unbounded memory growth under load.
pub struct ZmqTransport {
    config: ZmqTransportConfig,
    dealer: Mutex<dealer::Dealer>,
    pusher: Mutex<push::Push>,
}

impl ZmqTransport {
    /// Connect to a registry hub.
    pub fn connect(config: ZmqTransportConfig) -> Result<Self> {
        let context = Context::new();

        // Create DEALER socket with HWM
        let dealer = dealer::dealer(&context)
            .set_sndhwm(config.dealer_hwm)
            .set_rcvhwm(config.dealer_hwm)
            .connect(&config.query_addr)
            .map_err(|e| anyhow!("Failed to connect DEALER to {}: {}", config.query_addr, e))?;

        // Create PUSH socket with HWM
        let pusher = push::push(&context)
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
    pub fn connect_to_with_hwm(host: &str, query_port: u16, publish_port: u16, hwm: i32) -> Result<Self> {
        let config = ZmqTransportConfig::new(
            format!("tcp://{}:{}", host, query_port),
            format!("tcp://{}:{}", host, publish_port),
        ).with_hwm(hwm);
        Self::connect(config)
    }
}

#[async_trait]
impl RegistryTransport for ZmqTransport {
    async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut socket = self.dealer.lock().await;

        let mut msg = VecDeque::new();
        msg.push_back(Message::from(data.to_vec()));
        socket.send(Multipart(msg)).await
            .map_err(|e| anyhow!("Failed to send request: {}", e))?;

        let response = tokio::time::timeout(self.config.request_timeout, async {
            match socket.next().await {
                Some(Ok(msg)) => {
                    let frames: Vec<_> = msg.iter().collect();
                    if frames.is_empty() {
                        return Err(anyhow!("Empty response"));
                    }
                    Ok(frames[0].to_vec())
                }
                Some(Err(e)) => Err(anyhow!("Receive error: {}", e)),
                None => Err(anyhow!("Socket closed")),
            }
        })
        .await
        .map_err(|_| anyhow!("Request timed out after {:?}", self.config.request_timeout))??;

        Ok(response)
    }

    async fn publish(&self, data: &[u8]) -> Result<()> {
        let mut socket = self.pusher.lock().await;

        let mut msg = VecDeque::new();
        msg.push_back(Message::from(data.to_vec()));
        socket.send(Multipart(msg)).await
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
        let config = ZmqTransportConfig::new("tcp://host:1234", "tcp://host:5678")
            .with_timeout(Duration::from_secs(10))
            .with_hwm(5000);

        assert_eq!(config.query_addr, "tcp://host:1234");
        assert_eq!(config.publish_addr, "tcp://host:5678");
        assert_eq!(config.request_timeout, Duration::from_secs(10));
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
