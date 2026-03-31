// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unix domain socket hub transport for the distributed registry.
//!
//! Mirrors `velo_hub_transport.rs` but uses `tokio::net::UnixListener` and
//! `tokio::net::UnixStream` instead of TCP. The framing is identical —
//! `TcpFrameCodec` is reused since the codec is transport-agnostic.
//!
//! Wire protocol:
//! - `MessageType::Message`  → `HubMessage::Query`  (requires response)
//! - `MessageType::Ack`      → `HubMessage::Publish` (fire-and-forget)
//! - `MessageType::Response` → sent back to a specific client

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dashmap::DashMap;
use tokio::io::{AsyncWriteExt, WriteHalf};
use tokio::net::{UnixListener as TokioUnixListener, UnixStream};
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

// ── UdsHubTransport ────────────────────────────────────────────────────────

/// Unix domain socket hub transport.
///
/// All workers connect to a single UDS path. Queries use `MessageType::Message`
/// (hub responds via `MessageType::Response`); registrations use `MessageType::Ack`
/// (fire-and-forget).
pub struct UdsHubTransport {
    /// Incoming message channel (fed by per-connection reader tasks).
    rx: mpsc::UnboundedReceiver<Envelope>,
    /// Write halves for active connections, keyed by `ClientId`.
    writers: Arc<DashMap<ClientId, WriteHalf<UnixStream>>>,
    /// Background accept task handle.
    _accept_handle: tokio::task::JoinHandle<()>,
    /// Socket path (kept for cleanup).
    socket_path: Option<PathBuf>,
}

impl UdsHubTransport {
    /// Bind and start listening on the given UDS path.
    pub async fn bind(path: &Path) -> Result<Self> {
        // Remove stale socket if it exists
        if path.exists() {
            std::fs::remove_file(path).ok();
        }

        let listener = TokioUnixListener::bind(path)
            .map_err(|e| anyhow!("UdsHubTransport: failed to bind {:?}: {e}", path))?;

        tracing::info!(path = ?path, "UdsHubTransport bound");

        let (tx, rx) = mpsc::unbounded_channel::<Envelope>();
        let writers: Arc<DashMap<ClientId, WriteHalf<UnixStream>>> = Arc::new(DashMap::new());
        let writers_clone = writers.clone();

        let accept_handle = tokio::spawn(async move {
            Self::accept_loop(listener, tx, writers_clone).await;
        });

        Ok(Self {
            rx,
            writers,
            _accept_handle: accept_handle,
            socket_path: Some(path.to_path_buf()),
        })
    }

    /// Bind on a temporary socket file. Returns the transport and the path.
    pub async fn bind_temp() -> Result<(Self, PathBuf)> {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("kvbm-registry-{}.sock", std::process::id()));
        let transport = Self::bind(&path).await?;
        Ok((transport, path))
    }

    // ── Accept loop ──────────────────────────────────────────────────

    async fn accept_loop(
        listener: TokioUnixListener,
        tx: mpsc::UnboundedSender<Envelope>,
        writers: Arc<DashMap<ClientId, WriteHalf<UnixStream>>>,
    ) {
        static CLIENT_COUNTER: AtomicU64 = AtomicU64::new(1);

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let id = CLIENT_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let client_id = ClientId::new(id.to_le_bytes().to_vec());

                    let (read_half, write_half) = tokio::io::split(stream);
                    writers.insert(client_id.clone(), write_half);

                    let tx2 = tx.clone();
                    let writers2 = writers.clone();
                    tokio::spawn(async move {
                        Self::read_loop(read_half, client_id.clone(), tx2).await;
                        writers2.remove(&client_id);
                        tracing::debug!("UdsHubTransport: client disconnected");
                    });
                }
                Err(e) => {
                    tracing::error!(err = %e, "UdsHubTransport: accept error");
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
    }

    // ── Per-connection read loop ──────────────────────────────────────

    async fn read_loop(
        read_half: tokio::io::ReadHalf<UnixStream>,
        client_id: ClientId,
        tx: mpsc::UnboundedSender<Envelope>,
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
                        break;
                    }
                }
                Err(e) => {
                    tracing::warn!(err = %e, "UdsHubTransport: read error");
                    break;
                }
            }
        }
    }
}

impl Drop for UdsHubTransport {
    fn drop(&mut self) {
        if let Some(path) = self.socket_path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[async_trait]
impl HubTransport for UdsHubTransport {
    async fn recv(&mut self) -> Result<HubMessage> {
        loop {
            let envelope = self
                .rx
                .recv()
                .await
                .ok_or_else(|| anyhow!("UdsHubTransport: all clients disconnected"))?;

            return match envelope.msg_type {
                MessageType::Message => Ok(HubMessage::Query {
                    client: envelope.client,
                    data: envelope.payload,
                }),
                MessageType::Ack => Ok(HubMessage::Publish {
                    data: envelope.payload,
                }),
                MessageType::Response | MessageType::Event | MessageType::ShuttingDown => {
                    tracing::debug!(
                        msg_type = ?envelope.msg_type,
                        "UdsHubTransport: unexpected frame type from client, skipping"
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
            .ok_or_else(|| anyhow!("UdsHubTransport: client not found"))?;

        let mut frame = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut frame, MessageType::Response, &[], data)
            .map_err(|e| anyhow!("UdsHubTransport: encode error: {e}"))?;

        entry
            .write_all(&frame)
            .await
            .map_err(|e| anyhow!("UdsHubTransport: write error: {e}"))
    }

    fn name(&self) -> &'static str {
        "uds"
    }
}

// ── UdsClientTransport ─────────────────────────────────────────────────────

/// Client-side UDS transport that connects to a `UdsHubTransport`.
pub struct UdsClientTransport {
    stream: UnixStream,
    codec: TcpFrameCodec,
}

impl UdsClientTransport {
    /// Connect to a hub at the given UDS path.
    pub async fn connect(path: &Path) -> Result<Self> {
        let stream = UnixStream::connect(path)
            .await
            .map_err(|e| anyhow!("UdsClientTransport: failed to connect to {:?}: {e}", path))?;
        Ok(Self {
            stream,
            codec: TcpFrameCodec::new(),
        })
    }

    /// Send a request (query) and wait for the response.
    pub async fn request(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let mut frame = Vec::new();
        TcpFrameCodec::encode_frame_sync(&mut frame, MessageType::Message, &[], data)
            .map_err(|e| anyhow!("UdsClientTransport: encode error: {e}"))?;

        use tokio::io::AsyncWriteExt;
        self.stream.write_all(&frame).await?;

        use tokio::io::AsyncReadExt;
        let mut buf = Vec::with_capacity(4096);
        loop {
            let mut tmp = [0u8; 4096];
            let n = self.stream.read(&mut tmp).await?;
            if n == 0 {
                return Err(anyhow!("UdsClientTransport: connection closed"));
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
            .map_err(|e| anyhow!("UdsClientTransport: encode error: {e}"))?;
        use tokio::io::AsyncWriteExt;
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
    async fn test_uds_hub_roundtrip() {
        let (mut hub, path) = UdsHubTransport::bind_temp().await.unwrap();

        let path2 = path.clone();
        let handle = tokio::spawn(async move {
            let mut client = UdsClientTransport::connect(&path2).await.unwrap();
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
    async fn test_uds_hub_publish() {
        let (mut hub, path) = UdsHubTransport::bind_temp().await.unwrap();

        let path2 = path.clone();
        let handle = tokio::spawn(async move {
            let mut client = UdsClientTransport::connect(&path2).await.unwrap();
            client.publish(b"register-me").await.unwrap();
        });

        if let Ok(HubMessage::Publish { data }) = hub.recv().await {
            assert_eq!(data, b"register-me");
        }

        handle.await.unwrap();
    }
}
