// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap rendezvous for disaggregated mocker testing.
//!
//! Simulates the SGLang disaggregated serving handshake where prefill workers
//! wait for decode workers to connect before completing.
//!
//! - Prefill mockers bind a TCP listener and wait for decode to connect
//! - Decode mockers connect to prefill's bootstrap endpoint and send the room ID
//! - Both proceed after the handshake completes
//!
//! Wire protocol:
//! - Decode -> Prefill: room_id (8 bytes, little-endian u64)
//! - Prefill -> Decode: ACK (1 byte, 0x01)

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use dashmap::DashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Timeout for bootstrap rendezvous operations.
/// Hard fail on timeout for CI testing.
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// ACK byte sent from prefill to decode after successful room match.
const ACK_BYTE: u8 = 0x01;

/// Bootstrap server for prefill mockers.
/// One server per worker process, shared across all DP ranks.
pub struct BootstrapServer {
    port: u16,
    /// Maps room_id -> oneshot sender to unblock waiting prefill
    pending_rooms: Arc<DashMap<u64, oneshot::Sender<()>>>,
}

impl BootstrapServer {
    /// Start the bootstrap server on the specified port.
    ///
    /// Spawns a background task to accept incoming connections from decode workers.
    /// The server runs until the cancellation token is triggered.
    pub async fn start(port: u16, cancel_token: CancellationToken) -> Result<Arc<Self>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
        let actual_port = listener.local_addr()?.port();

        tracing::info!(port = actual_port, "Bootstrap server started");

        let pending_rooms: Arc<DashMap<u64, oneshot::Sender<()>>> = Arc::new(DashMap::new());
        let server = Arc::new(Self {
            port: actual_port,
            pending_rooms: pending_rooms.clone(),
        });

        // Spawn accept loop
        let rooms = pending_rooms;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, addr)) => {
                                tracing::debug!(peer = %addr, "Bootstrap connection accepted");
                                let rooms_clone = rooms.clone();
                                // Handle each connection in its own task
                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_connection(stream, rooms_clone).await {
                                        tracing::warn!(error = %e, "Bootstrap connection handling failed");
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Bootstrap accept failed");
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

    /// Handle a single connection from a decode worker.
    async fn handle_connection(
        mut stream: TcpStream,
        rooms: Arc<DashMap<u64, oneshot::Sender<()>>>,
    ) -> Result<()> {
        // Read room_id (8 bytes, little-endian)
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf).await?;
        let room_id = u64::from_le_bytes(buf);

        tracing::debug!(
            room_id = room_id,
            "Bootstrap: decode connected with room_id"
        );

        // Find and notify waiting prefill
        if let Some((_, sender)) = rooms.remove(&room_id) {
            // Notify prefill that decode has connected
            let _ = sender.send(());
            // Send ACK to decode
            stream.write_all(&[ACK_BYTE]).await?;
            tracing::debug!(room_id = room_id, "Bootstrap: room matched, ACK sent");
        } else {
            tracing::warn!(
                room_id = room_id,
                "Bootstrap: no pending prefill for room_id"
            );
            // Still send ACK to not hang the decode worker
            stream.write_all(&[ACK_BYTE]).await?;
        }

        Ok(())
    }

    /// Wait for a decode worker to connect with the given room_id.
    ///
    /// Called by prefill mocker when processing a request with bootstrap_info.
    /// Times out after 30 seconds and returns an error (hard fail for CI).
    pub async fn wait_for_room(&self, room_id: u64) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.pending_rooms.insert(room_id, tx);

        tracing::debug!(
            room_id = room_id,
            port = self.port,
            "Bootstrap: prefill waiting for decode"
        );

        match tokio::time::timeout(RENDEZVOUS_TIMEOUT, rx).await {
            Ok(Ok(())) => {
                tracing::debug!(room_id = room_id, "Bootstrap: prefill unblocked by decode");
                Ok(())
            }
            Ok(Err(_)) => {
                // Sender was dropped without sending - shouldn't happen
                self.pending_rooms.remove(&room_id);
                bail!("Bootstrap rendezvous cancelled for room {}", room_id)
            }
            Err(_) => {
                // Timeout
                self.pending_rooms.remove(&room_id);
                bail!(
                    "Bootstrap rendezvous timed out after {:?} for room {}",
                    RENDEZVOUS_TIMEOUT,
                    room_id
                )
            }
        }
    }

    /// Get the port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Connect to a prefill worker's bootstrap server and complete the rendezvous.
///
/// Called by decode mocker when processing a request with bootstrap_info.
/// Times out after 30 seconds and returns an error (hard fail for CI).
pub async fn connect_to_prefill(host: &str, port: u16, room_id: u64) -> Result<()> {
    // Strip brackets from IPv6 addresses if present
    let host = host.trim_matches(|c| c == '[' || c == ']');
    let addr = format!("{}:{}", host, port);

    tracing::debug!(
        addr = %addr,
        room_id = room_id,
        "Bootstrap: decode connecting to prefill"
    );

    // Connect with timeout
    let mut stream = tokio::time::timeout(RENDEZVOUS_TIMEOUT, TcpStream::connect(&addr))
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Bootstrap: failed to connect to prefill at {} - timeout after {:?}",
                addr,
                RENDEZVOUS_TIMEOUT
            )
        })?
        .map_err(|e| {
            anyhow::anyhow!("Bootstrap: failed to connect to prefill at {}: {}", addr, e)
        })?;

    // Send room_id (8 bytes, little-endian)
    stream.write_all(&room_id.to_le_bytes()).await?;

    // Wait for ACK with timeout
    let mut ack = [0u8; 1];
    tokio::time::timeout(RENDEZVOUS_TIMEOUT, stream.read_exact(&mut ack))
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Bootstrap: ACK timeout after {:?} for room {}",
                RENDEZVOUS_TIMEOUT,
                room_id
            )
        })?
        .map_err(|e| anyhow::anyhow!("Bootstrap: failed to read ACK: {}", e))?;

    if ack[0] != ACK_BYTE {
        bail!(
            "Bootstrap: invalid ACK byte {:?} for room {}",
            ack[0],
            room_id
        );
    }

    tracing::debug!(
        room_id = room_id,
        "Bootstrap: decode received ACK from prefill"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bootstrap_rendezvous() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .expect("Failed to start bootstrap server");

        let port = server.port();
        let room_id = 12345u64;

        // Spawn prefill waiting task
        let server_clone = server.clone();
        let prefill_handle = tokio::spawn(async move { server_clone.wait_for_room(room_id).await });

        // Give prefill a moment to register the room
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Decode connects
        let decode_result = connect_to_prefill("127.0.0.1", port, room_id).await;
        assert!(
            decode_result.is_ok(),
            "Decode connection failed: {:?}",
            decode_result
        );

        // Prefill should complete
        let prefill_result = prefill_handle.await.unwrap();
        assert!(
            prefill_result.is_ok(),
            "Prefill wait failed: {:?}",
            prefill_result
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_bootstrap_timeout() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .expect("Failed to start bootstrap server");

        let room_id = 99999u64;

        // Wait for a room that no one will connect to - should timeout
        // Use a shorter timeout for testing by checking the error message
        let result =
            tokio::time::timeout(Duration::from_millis(100), server.wait_for_room(room_id)).await;

        // Should timeout
        assert!(result.is_err(), "Expected timeout");

        cancel_token.cancel();
    }
}
