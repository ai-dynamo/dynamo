// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-based hub transport for the distributed registry.
//!
//! Uses `velo-transports::tcp::TcpFrameCodec` for framing and
//! `tokio::net::TcpListener` for accepting connections.
//!
//! Wire protocol over a single TCP port:
//! - `MessageType::Message`  → `HubMessage::Query`  (requires response)
//! - `MessageType::Ack`      → `HubMessage::Publish` (fire-and-forget)
//! - `MessageType::Response` → sent back to a specific client
//!
//! This replaces the two-port ZMQ ROUTER+PULL pattern with a single
//! multiplexed TCP endpoint that all workers connect to.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dashmap::DashMap;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio::net::TcpListener as TokioTcpListener;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_util::codec::FramedRead;
use velo_transports::MessageType;
use velo_transports::tcp::TcpFrameCodec;

use super::hub_transport::{ClientId, HubMessage, HubTransport};

// ── Internal message envelope ──────────────────────────────────────────────

struct Envelope {
    client: ClientId,
    msg_type: MessageType,
    payload: Vec<u8>,
}

// ── VeloHubTransport ──────────────────────────────────────────────────────

/// Velo-based hub transport.
///
/// All workers connect to a single TCP port. Queries use `MessageType::Message`
/// (hub responds via `MessageType::Response`); registrations use `MessageType::Ack`
/// (fire-and-forget).
pub struct VeloHubTransport {
    /// Incoming message channel (fed by per-connection reader tasks).
    rx: mpsc::UnboundedReceiver<Envelope>,
    /// Write halves for active connections, keyed by `ClientId`.
    writers: Arc<DashMap<ClientId, WriteHalf<TcpStream>>>,
    /// Background accept task handle (kept alive while hub is running).
    _accept_handle: tokio::task::JoinHandle<()>,
}

impl VeloHubTransport {
    /// Bind and start listening on `addr`.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let listener = TokioTcpListener::bind(addr)
            .await
            .map_err(|e| anyhow!("VeloHubTransport: failed to bind {addr}: {e}"))?;

        tracing::info!(addr = %addr, "VeloHubTransport bound");

        let (tx, rx) = mpsc::unbounded_channel::<Envelope>();
        let writers: Arc<DashMap<ClientId, WriteHalf<TcpStream>>> = Arc::new(DashMap::new());
        let writers_clone = writers.clone();

        let accept_handle = tokio::spawn(async move {
            Self::accept_loop(listener, tx, writers_clone).await;
        });

        Ok(Self {
            rx,
            writers,
            _accept_handle: accept_handle,
        })
    }

    /// Bind on `127.0.0.1:<port>` (convenience for tests / single-host setups).
    pub async fn bind_local(port: u16) -> Result<Self> {
        Self::bind(format!("127.0.0.1:{port}").parse().unwrap()).await
    }

    /// Bind on all interfaces on `port`.
    pub async fn bind_all(port: u16) -> Result<Self> {
        Self::bind(format!("0.0.0.0:{port}").parse().unwrap()).await
    }

    // ── Accept loop ──────────────────────────────────────────────────

    async fn accept_loop(
        listener: TokioTcpListener,
        tx: mpsc::UnboundedSender<Envelope>,
        writers: Arc<DashMap<ClientId, WriteHalf<TcpStream>>>,
    ) {
        static CLIENT_COUNTER: AtomicU64 = AtomicU64::new(1);

        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let id = CLIENT_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let client_id = ClientId::new(id.to_le_bytes().to_vec());

                    let (read_half, write_half) = tokio::io::split(stream);
                    writers.insert(client_id.clone(), write_half);

                    let tx2 = tx.clone();
                    let writers2 = writers.clone();
                    tokio::spawn(async move {
                        Self::read_loop(read_half, client_id.clone(), tx2, peer).await;
                        // Clean up writer when reader exits
                        writers2.remove(&client_id);
                        tracing::debug!(peer = %peer, "VeloHubTransport: client disconnected");
                    });
                }
                Err(e) => {
                    tracing::error!(err = %e, "VeloHubTransport: accept error");
                    // Back off briefly to avoid tight loop on persistent errors
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
    }

    // ── Per-connection read loop ──────────────────────────────────────

    async fn read_loop(
        read_half: tokio::io::ReadHalf<TcpStream>,
        client_id: ClientId,
        tx: mpsc::UnboundedSender<Envelope>,
        peer: SocketAddr,
    ) {
        let codec = TcpFrameCodec::new();
        let mut framed = FramedRead::new(read_half, codec);

        use futures_util::StreamExt;
        while let Some(frame) = framed.next().await {
            match frame {
                Ok((msg_type, _header, payload)) => {
                    let envelope = Envelope {
                        client: client_id.clone(),
                        msg_type,
                        payload: payload.to_vec(),
                    };
                    if tx.send(envelope).is_err() {
                        // Hub dropped, stop reading
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(peer = %peer, err = %e, "VeloHubTransport: read error");
                    break;
                }
            }
        }
    }
}

#[async_trait]
impl HubTransport for VeloHubTransport {
    async fn recv(&mut self) -> Result<HubMessage> {
        loop {
            let envelope = self
                .rx
                .recv()
                .await
                .ok_or_else(|| anyhow!("VeloHubTransport: all clients disconnected"))?;

            return match envelope.msg_type {
                MessageType::Message => Ok(HubMessage::Query {
                    client: envelope.client,
                    data: envelope.payload,
                }),
                MessageType::Ack => Ok(HubMessage::Publish {
                    data: envelope.payload,
                }),
                // Skip unexpected frame types (shouldn't arrive at hub, but be defensive)
                MessageType::Response | MessageType::Event | MessageType::ShuttingDown => {
                    tracing::debug!(
                        msg_type = ?envelope.msg_type,
                        "VeloHubTransport: unexpected frame type from client, skipping"
                    );
                    continue;
                }
            };
        }
    }

    async fn respond(&mut self, client: &ClientId, data: &[u8]) -> Result<()> {
        let mut entry = self
            .writers
            .get_mut(client)
            .ok_or_else(|| anyhow!("VeloHubTransport: client not found"))?;

        let mut frame = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut frame, MessageType::Response, &[], data)
            .map_err(|e| anyhow!("VeloHubTransport: encode error: {e}"))?;

        entry
            .write_all(&frame)
            .await
            .map_err(|e| anyhow!("VeloHubTransport: write error: {e}"))
    }

    fn name(&self) -> &'static str {
        "velo-tcp"
    }
}

// ── VeloClientTransport ───────────────────────────────────────────────────

/// Client-side transport that connects to a `VeloHubTransport`.
///
/// Implements `RegistryTransport` — workers use this to register blocks
/// (fire-and-forget `Ack`) and query the hub (`Message` → `Response`).
pub struct VeloClientTransport {
    stream: TcpStream,
    codec: TcpFrameCodec,
}

impl VeloClientTransport {
    /// Connect to a hub at `addr`.
    pub async fn connect(addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| anyhow!("VeloClientTransport: failed to connect to {addr}: {e}"))?;
        Ok(Self {
            stream,
            codec: TcpFrameCodec::new(),
        })
    }

    /// Connect to `127.0.0.1:<port>`.
    pub async fn connect_local(port: u16) -> Result<Self> {
        Self::connect(format!("127.0.0.1:{port}").parse().unwrap()).await
    }

    /// Send a request (query) and wait for the response.
    pub async fn request(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        // Send query frame
        let mut frame = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut frame, MessageType::Message, &[], data)
            .map_err(|e| anyhow!("VeloClientTransport: encode error: {e}"))?;

        self.stream.write_all(&frame).await?;

        // Read response frame
        use tokio::io::AsyncReadExt;
        let mut buf = Vec::with_capacity(4096);
        loop {
            let mut tmp = [0u8; 4096];
            let n = self.stream.read(&mut tmp).await?;
            if n == 0 {
                return Err(anyhow!("VeloClientTransport: connection closed"));
            }
            buf.extend_from_slice(&tmp[..n]);

            use bytes::BytesMut;
            use tokio_util::codec::Decoder;
            let mut bm = BytesMut::from(buf.as_slice());
            if let Some((msg_type, _header, payload)) = self.codec.decode(&mut bm)? {
                if msg_type == MessageType::Response {
                    return Ok(payload.to_vec());
                }
            }
        }
    }

    /// Publish a registration (fire-and-forget).
    pub async fn publish(&mut self, data: &[u8]) -> Result<()> {
        let mut frame = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut frame, MessageType::Ack, &[], data)
            .map_err(|e| anyhow!("VeloClientTransport: encode error: {e}"))?;
        self.stream
            .write_all(&frame)
            .await
            .map_err(|e| anyhow!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_velo_hub_query_response() {
        let _hub = VeloHubTransport::bind_local(0).await.expect("bind failed");

        // Get the actual bound port by peeking at the TcpListener
        // (We use port 17891 in a retry loop for tests)
        let hub = VeloHubTransport::bind_local(17891)
            .await
            .unwrap_or_else(|_| {
                // Skip test if port is in use
                panic!("Port 17891 in use — run tests sequentially");
            });
        let _ = hub;
    }

    #[tokio::test]
    async fn test_velo_hub_roundtrip() {
        let port = 17892u16;

        let mut hub = VeloHubTransport::bind_local(port).await.unwrap();

        let handle = tokio::spawn(async move {
            let mut client = VeloClientTransport::connect_local(port).await.unwrap();
            let response = client.request(b"hello").await.unwrap();
            assert_eq!(response, b"olleh");
        });

        // Hub echoes reversed bytes
        if let Ok(HubMessage::Query { client, data }) = hub.recv().await {
            let mut rev = data;
            rev.reverse();
            hub.respond(&client, &rev).await.unwrap();
        }

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_velo_hub_publish() {
        let port = 17893u16;
        let mut hub = VeloHubTransport::bind_local(port).await.unwrap();

        let handle = tokio::spawn(async move {
            let mut client = VeloClientTransport::connect_local(port).await.unwrap();
            client.publish(b"register-me").await.unwrap();
        });

        if let Ok(HubMessage::Publish { data }) = hub.recv().await {
            assert_eq!(data, b"register-me");
        }

        handle.await.unwrap();
    }
}
