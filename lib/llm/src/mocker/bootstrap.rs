// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap rendezvous for disaggregated mocker testing.
//!
//! Simulates the SGLang disaggregated serving handshake for KV transfer coordination.
//! Either prefill or decode can arrive first; the rendezvous completes when both are ready.
//!
//! - Prefill: calls `complete_room(room_id)` after first token (KV cache ready)
//! - Decode: connects to prefill's bootstrap server, blocks until prefill completes
//!
//! Wire protocol:
//! - Decode -> Prefill: room_id (8 bytes, little-endian u64)
//! - Prefill -> Decode: ACK (1 byte, 0x01) after prefill completes

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Timeout for bootstrap rendezvous operations.
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// ACK byte sent from server to decode after prefill completes.
const ACK_BYTE: u8 = 0x01;

/// State for a room in the rendezvous.
struct RoomState {
    /// True if prefill has completed (KV cache ready)
    prefill_completed: bool,
    /// Channel to notify decode when prefill completes (if decode is waiting)
    decode_waiting: Option<oneshot::Sender<()>>,
}

/// Bootstrap server for prefill mockers.
/// Handles rendezvous between prefill and decode for KV transfer coordination.
pub struct BootstrapServer {
    port: u16,
    rooms: Arc<DashMap<u64, RoomState>>,
}

impl BootstrapServer {
    /// Start the bootstrap server on the specified port.
    pub async fn start(port: u16, cancel_token: CancellationToken) -> Result<Arc<Self>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
        let actual_port = listener.local_addr()?.port();

        tracing::info!("Bootstrap server started on port {actual_port}");

        let rooms: Arc<DashMap<u64, RoomState>> = Arc::new(DashMap::new());
        let server = Arc::new(Self {
            port: actual_port,
            rooms: rooms.clone(),
        });

        // Spawn accept loop
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, addr)) => {
                                tracing::debug!("Bootstrap: accepted connection from {addr}");
                                let rooms_clone = rooms.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_connection(stream, rooms_clone).await {
                                        tracing::warn!("Bootstrap: connection error: {e}");
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::warn!("Bootstrap: accept failed: {e}");
                            }
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("Bootstrap server shutting down");
                        break;
                    }
                }
            }
        });

        Ok(server)
    }

    /// Handle a connection from decode. Blocks until prefill completes for this room.
    async fn handle_connection(
        mut stream: TcpStream,
        rooms: Arc<DashMap<u64, RoomState>>,
    ) -> Result<()> {
        // Read room_id (8 bytes, little-endian)
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf).await?;
        let room_id = u64::from_le_bytes(buf);

        tracing::debug!("Bootstrap: decode connected for room {room_id}");

        // Check room state and wait if needed
        let rx = match rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                if entry.get().prefill_completed {
                    // Prefill already done, immediate ACK
                    entry.remove();
                    tracing::debug!("Bootstrap: room {room_id} already completed, immediate ACK");
                    None
                } else {
                    // Prefill registered but not completed, wait
                    let (tx, rx) = oneshot::channel();
                    entry.get_mut().decode_waiting = Some(tx);
                    tracing::debug!("Bootstrap: room {room_id} waiting for prefill to complete");
                    Some(rx)
                }
            }
            Entry::Vacant(entry) => {
                // Decode arrived first, create entry and wait
                let (tx, rx) = oneshot::channel();
                entry.insert(RoomState {
                    prefill_completed: false,
                    decode_waiting: Some(tx),
                });
                tracing::debug!("Bootstrap: room {room_id} decode arrived first, waiting");
                Some(rx)
            }
        };

        // Wait for prefill to complete if needed
        if let Some(rx) = rx {
            match tokio::time::timeout(RENDEZVOUS_TIMEOUT, rx).await {
                Ok(Ok(())) => {
                    tracing::debug!("Bootstrap: room {room_id} prefill completed, sending ACK");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {room_id} sender dropped");
                }
                Err(_) => {
                    rooms.remove(&room_id);
                    bail!("Bootstrap: room {room_id} timeout waiting for prefill");
                }
            }
        }

        // Send ACK
        stream.write_all(&[ACK_BYTE]).await?;
        Ok(())
    }

    /// Mark a room as completed (prefill finished, KV cache ready).
    /// If decode is already waiting, unblocks it.
    pub fn complete_room(&self, room_id: u64) {
        match self.rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                if let Some(sender) = entry.get_mut().decode_waiting.take() {
                    // Decode is waiting, unblock it
                    let _ = sender.send(());
                    entry.remove();
                    tracing::debug!("Bootstrap: room {room_id} completed, decode unblocked");
                } else {
                    // Decode not connected yet, mark completed
                    entry.get_mut().prefill_completed = true;
                    tracing::debug!("Bootstrap: room {room_id} completed, awaiting decode");
                }
            }
            Entry::Vacant(entry) => {
                // Decode hasn't connected yet
                entry.insert(RoomState {
                    prefill_completed: true,
                    decode_waiting: None,
                });
                tracing::debug!("Bootstrap: room {room_id} completed (no decode yet)");
            }
        }
    }

    /// Get the port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Connect to a prefill worker's bootstrap server and wait for KV to be ready.
pub async fn connect_to_prefill(host: &str, port: u16, room_id: u64) -> Result<()> {
    let host = host.trim_matches(|c| c == '[' || c == ']');
    let addr = format!("{host}:{port}");

    tracing::debug!("Bootstrap: decode connecting to {addr} for room {room_id}");

    // Connect with timeout
    let mut stream = tokio::time::timeout(RENDEZVOUS_TIMEOUT, TcpStream::connect(&addr))
        .await
        .map_err(|_| anyhow::anyhow!("Bootstrap: connect timeout to {addr}"))?
        .map_err(|e| anyhow::anyhow!("Bootstrap: connect failed to {addr}: {e}"))?;

    // Send room_id
    stream.write_all(&room_id.to_le_bytes()).await?;

    // Wait for ACK (blocks until prefill completes)
    let mut ack = [0u8; 1];
    tokio::time::timeout(RENDEZVOUS_TIMEOUT, stream.read_exact(&mut ack))
        .await
        .map_err(|_| anyhow::anyhow!("Bootstrap: ACK timeout for room {room_id}"))?
        .map_err(|e| anyhow::anyhow!("Bootstrap: read ACK failed: {e}"))?;

    if ack[0] != ACK_BYTE {
        bail!(
            "Bootstrap: invalid ACK byte {:02x} for room {room_id}",
            ack[0]
        );
    }

    tracing::debug!("Bootstrap: decode received ACK for room {room_id}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefill_completes_first() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 1001u64;

        // Prefill completes first
        server.complete_room(room_id);

        // Decode connects - should get immediate ACK
        let result = connect_to_prefill("127.0.0.1", port, room_id).await;
        assert!(result.is_ok(), "Decode should succeed: {result:?}");

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_decode_connects_first() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 1002u64;

        // Spawn decode (will block waiting for prefill)
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });

        // Give decode time to connect and register
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Prefill completes - should unblock decode
        server.complete_room(room_id);

        let result = decode_handle.await.unwrap();
        assert!(result.is_ok(), "Decode should succeed: {result:?}");

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_interleaved_ordering() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 1003u64;

        // Spawn decode
        let server_clone = server.clone();
        let decode_handle = tokio::spawn(async move {
            // Small delay so prefill can "register" conceptually first
            tokio::time::sleep(Duration::from_millis(10)).await;
            connect_to_prefill("127.0.0.1", port, room_id).await
        });

        // Prefill completes after decode starts connecting
        tokio::time::sleep(Duration::from_millis(50)).await;
        server_clone.complete_room(room_id);

        let result = decode_handle.await.unwrap();
        assert!(result.is_ok(), "Decode should succeed: {result:?}");

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_multiple_rooms_concurrent() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();

        let mut handles = vec![];

        // Room 1: prefill first
        let server1 = server.clone();
        handles.push(tokio::spawn(async move {
            server1.complete_room(2001);
            tokio::time::sleep(Duration::from_millis(10)).await;
            connect_to_prefill("127.0.0.1", port, 2001).await
        }));

        // Room 2: decode first
        let server2 = server.clone();
        handles.push(tokio::spawn(async move {
            let decode = tokio::spawn(connect_to_prefill("127.0.0.1", port, 2002));
            tokio::time::sleep(Duration::from_millis(50)).await;
            server2.complete_room(2002);
            decode.await.unwrap()
        }));

        // Room 3: simultaneous
        let server3 = server.clone();
        handles.push(tokio::spawn(async move {
            let decode = tokio::spawn(connect_to_prefill("127.0.0.1", port, 2003));
            server3.complete_room(2003);
            decode.await.unwrap()
        }));

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.unwrap();
            assert!(
                result.is_ok(),
                "Room {} should succeed: {result:?}",
                2001 + i
            );
        }

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_decode_timeout_no_prefill() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 9999u64;

        // Decode connects but prefill never completes - use short timeout
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            connect_to_prefill("127.0.0.1", port, room_id),
        )
        .await;

        // Should timeout (outer timeout, not inner RENDEZVOUS_TIMEOUT)
        assert!(result.is_err(), "Should timeout waiting for prefill");

        cancel_token.cancel();
    }
}
