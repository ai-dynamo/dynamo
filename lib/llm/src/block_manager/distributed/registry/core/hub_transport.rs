// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-side transport traits and implementations.

use std::collections::VecDeque;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use tmq::{Context, Message, Multipart, pull, router};
use tokio::sync::mpsc;

/// Opaque client identifier for routing responses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClientId(Vec<u8>);

impl ClientId {
    pub fn new(id: Vec<u8>) -> Self {
        Self(id)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Message received by the hub.
#[derive(Debug)]
pub enum HubMessage {
    /// Query from a client (needs response).
    Query { client: ClientId, data: Vec<u8> },
    /// Published registration (fire-and-forget).
    Publish { data: Vec<u8> },
}

/// Server-side transport for the registry hub.
///
/// Note: Not `Sync` because ZMQ sockets can't be shared across threads.
/// The hub uses `&mut self` and is owned by a single serve task.
#[async_trait]
pub trait HubTransport: Send {
    /// Receive the next message (query or publish).
    async fn recv(&mut self) -> Result<HubMessage>;

    /// Send a response to a specific client.
    async fn respond(&mut self, client: &ClientId, data: &[u8]) -> Result<()>;

    /// Get the transport name.
    fn name(&self) -> &'static str;
}

/// In-process hub transport for testing.
pub struct InProcessHubTransport {
    query_rx: mpsc::UnboundedReceiver<(ClientId, Vec<u8>)>,
    query_response_tx: mpsc::UnboundedSender<(ClientId, Vec<u8>)>,
    publish_rx: mpsc::UnboundedReceiver<Vec<u8>>,
}

/// Handle for clients to communicate with InProcessHubTransport.
pub struct InProcessClientHandle {
    query_tx: mpsc::UnboundedSender<(ClientId, Vec<u8>)>,
    query_response_rx: Mutex<mpsc::UnboundedReceiver<(ClientId, Vec<u8>)>>,
    publish_tx: mpsc::UnboundedSender<Vec<u8>>,
    client_id: ClientId,
}

impl InProcessHubTransport {
    /// Create a new in-process hub transport and client handle.
    pub fn new() -> (Self, InProcessClientHandle) {
        let (query_tx, query_rx) = mpsc::unbounded_channel();
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let (publish_tx, publish_rx) = mpsc::unbounded_channel();

        let transport = Self {
            query_rx,
            query_response_tx: response_tx,
            publish_rx,
        };

        let handle = InProcessClientHandle {
            query_tx,
            query_response_rx: Mutex::new(response_rx),
            publish_tx,
            client_id: ClientId::new(vec![0, 1, 2, 3]),
        };

        (transport, handle)
    }
}

impl Default for InProcessHubTransport {
    fn default() -> Self {
        Self::new().0
    }
}

impl InProcessClientHandle {
    /// Send a query and wait for response.
    pub async fn request(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.query_tx
            .send((self.client_id.clone(), data.to_vec()))
            .map_err(|_| anyhow!("Hub disconnected"))?;

        let mut rx = self.query_response_rx.lock();
        match rx.recv().await {
            Some((_, response)) => Ok(response),
            None => Err(anyhow!("Hub disconnected")),
        }
    }

    /// Publish a message (fire-and-forget).
    pub fn publish(&self, data: &[u8]) -> Result<()> {
        self.publish_tx
            .send(data.to_vec())
            .map_err(|_| anyhow!("Hub disconnected"))
    }
}

#[async_trait]
impl HubTransport for InProcessHubTransport {
    async fn recv(&mut self) -> Result<HubMessage> {
        tokio::select! {
            Some((client, data)) = self.query_rx.recv() => {
                Ok(HubMessage::Query { client, data })
            }
            Some(data) = self.publish_rx.recv() => {
                Ok(HubMessage::Publish { data })
            }
            else => Err(anyhow!("All clients disconnected"))
        }
    }

    async fn respond(&mut self, client: &ClientId, data: &[u8]) -> Result<()> {
        self.query_response_tx
            .send((client.clone(), data.to_vec()))
            .map_err(|_| anyhow!("Client disconnected"))
    }

    fn name(&self) -> &'static str {
        "in_process"
    }
}

/// ZMQ-based hub transport.
///
/// Uses ROUTER for queries and PULL for registrations.
pub struct ZmqHubTransport {
    router: router::Router,
    puller: pull::Pull,
}

/// Default high-water mark for ZMQ hub sockets.
pub const DEFAULT_HUB_HWM: i32 = 10_000;

/// Configuration for ZMQ hub transport.
#[derive(Clone, Debug)]
pub struct ZmqHubConfig {
    pub query_bind_addr: String,
    pub subscribe_bind_addr: String,
    /// High-water mark for the ROUTER (query) socket.
    pub router_hwm: i32,
    /// High-water mark for the PULL (registration) socket.
    pub pull_hwm: i32,
}

impl ZmqHubConfig {
    pub fn new(query_addr: impl Into<String>, subscribe_addr: impl Into<String>) -> Self {
        Self {
            query_bind_addr: query_addr.into(),
            subscribe_bind_addr: subscribe_addr.into(),
            router_hwm: DEFAULT_HUB_HWM,
            pull_hwm: DEFAULT_HUB_HWM,
        }
    }

    pub fn default_ports(port_base: u16) -> Self {
        Self {
            query_bind_addr: format!("tcp://*:{}", port_base),
            subscribe_bind_addr: format!("tcp://*:{}", port_base + 1),
            router_hwm: DEFAULT_HUB_HWM,
            pull_hwm: DEFAULT_HUB_HWM,
        }
    }

    /// Set high-water marks for both sockets.
    pub fn with_hwm(mut self, hwm: i32) -> Self {
        self.router_hwm = hwm;
        self.pull_hwm = hwm;
        self
    }
}

impl Default for ZmqHubConfig {
    fn default() -> Self {
        Self::default_ports(5555)
    }
}

impl ZmqHubTransport {
    /// Bind and start listening.
    pub fn bind(config: ZmqHubConfig) -> Result<Self> {
        let context = Context::new();

        let router = router::router(&context)
            .set_sndhwm(config.router_hwm)
            .set_rcvhwm(config.router_hwm)
            .bind(&config.query_bind_addr)
            .map_err(|e| anyhow!("Failed to bind ROUTER to {}: {}", config.query_bind_addr, e))?;

        let puller = pull::pull(&context)
            .set_rcvhwm(config.pull_hwm)
            .bind(&config.subscribe_bind_addr)
            .map_err(|e| {
                anyhow!(
                    "Failed to bind PULL to {}: {}",
                    config.subscribe_bind_addr,
                    e
                )
            })?;

        tracing::info!(
            query_addr = %config.query_bind_addr,
            pull_addr = %config.subscribe_bind_addr,
            router_hwm = config.router_hwm,
            pull_hwm = config.pull_hwm,
            "ZMQ hub transport bound"
        );

        Ok(Self { router, puller })
    }

    /// Bind with default configuration.
    pub fn bind_default() -> Result<Self> {
        Self::bind(ZmqHubConfig::default())
    }
}

#[async_trait]
impl HubTransport for ZmqHubTransport {
    async fn recv(&mut self) -> Result<HubMessage> {
        tokio::select! {
            result = self.router.next() => {
                match result {
                    Some(Ok(msg)) => {
                        let frames: Vec<_> = msg.iter().collect();
                        if frames.len() < 2 {
                            return Err(anyhow!("Invalid ROUTER message: expected identity + data"));
                        }
                        let client = ClientId::new(frames[0].to_vec());
                        let data = frames[frames.len() - 1].to_vec();
                        Ok(HubMessage::Query { client, data })
                    }
                    Some(Err(e)) => Err(anyhow!("ROUTER receive error: {}", e)),
                    None => Err(anyhow!("ROUTER socket closed")),
                }
            }
            result = self.puller.next() => {
                match result {
                    Some(Ok(msg)) => {
                        let frames: Vec<_> = msg.iter().collect();
                        if frames.is_empty() {
                            return Err(anyhow!("Empty PULL message"));
                        }
                        let data = frames[0].to_vec();
                        Ok(HubMessage::Publish { data })
                    }
                    Some(Err(e)) => Err(anyhow!("PULL receive error: {}", e)),
                    None => Err(anyhow!("PULL socket closed")),
                }
            }
        }
    }

    async fn respond(&mut self, client: &ClientId, data: &[u8]) -> Result<()> {
        let mut msg = VecDeque::new();
        msg.push_back(Message::from(client.as_bytes().to_vec()));
        msg.push_back(Message::from(data.to_vec()));

        self.router
            .send(Multipart(msg))
            .await
            .map_err(|e| anyhow!("Failed to send response: {}", e))
    }

    fn name(&self) -> &'static str {
        "zmq"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_process_hub_transport() {
        let (mut transport, handle) = InProcessHubTransport::new();

        // Spawn hub handler
        let hub_task = tokio::spawn(async move {
            if let Ok(HubMessage::Query { client, data }) = transport.recv().await {
                let mut response = data;
                response.reverse();
                transport.respond(&client, &response).await.unwrap();
            }
        });

        // Client sends request
        let response = handle.request(b"hello").await.unwrap();
        assert_eq!(response, b"olleh");

        hub_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_in_process_publish() {
        let (mut transport, handle) = InProcessHubTransport::new();

        // Spawn hub handler
        let hub_task = tokio::spawn(async move {
            if let Ok(HubMessage::Publish { data }) = transport.recv().await {
                assert_eq!(data, b"registration");
            }
        });

        // Client publishes
        handle.publish(b"registration").unwrap();

        hub_task.await.unwrap();
    }
}
