// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap rendezvous for disaggregated mocker testing.
//!
//! Simulates the SGLang disaggregated serving handshake for KV transfer coordination.
//! Either prefill or decode can arrive first; prefill waits for decode metadata before
//! emitting output, and decode waits for prefill completion before generating.
//!
//! - Prefill: waits for decode metadata, then calls `complete_room(room_id)` after first token
//! - Decode: connects to prefill's bootstrap server, sends metadata, then waits for completion
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
    /// True if decode has sent receiver metadata for this room
    decode_ready: bool,
    /// True if prefill has completed (KV cache ready)
    prefill_completed: bool,
    /// Channel to notify prefill when decode metadata arrives
    prefill_waiting: Option<oneshot::Sender<()>>,
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

    /// Handle a connection from decode. Marks decode ready, then blocks until prefill completes.
    async fn handle_connection(
        mut stream: TcpStream,
        rooms: Arc<DashMap<u64, RoomState>>,
    ) -> Result<()> {
        // Read room_id (8 bytes, little-endian)
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf).await?;
        let room_id = u64::from_le_bytes(buf);

        tracing::debug!("Bootstrap: decode connected for room {room_id}");

        // Register decode metadata, wake prefill if it is waiting, then wait for prefill completion.
        let rx = match rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().decode_ready = true;
                if let Some(sender) = entry.get_mut().prefill_waiting.take() {
                    let _ = sender.send(());
                    tracing::debug!("Bootstrap: room {room_id} decode metadata unblocked prefill");
                }

                if entry.get().prefill_completed {
                    // Prefill already done, immediate ACK
                    entry.remove();
                    tracing::debug!("Bootstrap: room {room_id} already completed, immediate ACK");
                    None
                } else {
                    // Decode metadata is registered, but prefill has not completed yet.
                    let (tx, rx) = oneshot::channel();
                    entry.get_mut().decode_waiting = Some(tx);
                    tracing::debug!("Bootstrap: room {room_id} decode waiting for prefill");
                    Some(rx)
                }
            }
            Entry::Vacant(entry) => {
                // Decode arrived first, record metadata and wait for prefill completion.
                let (tx, rx) = oneshot::channel();
                entry.insert(RoomState {
                    decode_ready: true,
                    prefill_completed: false,
                    prefill_waiting: None,
                    decode_waiting: Some(tx),
                });
                tracing::debug!("Bootstrap: room {room_id} decode arrived first");
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

    /// Wait until decode has sent receiver metadata for this room.
    pub async fn wait_for_decode_ready(&self, room_id: u64) -> Result<()> {
        let rx = match self.rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                if entry.get().decode_ready {
                    tracing::debug!("Bootstrap: room {room_id} decode already ready");
                    None
                } else {
                    let (tx, rx) = oneshot::channel();
                    entry.get_mut().prefill_waiting = Some(tx);
                    tracing::debug!(
                        "Bootstrap: room {room_id} prefill waiting for decode metadata"
                    );
                    Some(rx)
                }
            }
            Entry::Vacant(entry) => {
                let (tx, rx) = oneshot::channel();
                entry.insert(RoomState {
                    decode_ready: false,
                    prefill_completed: false,
                    prefill_waiting: Some(tx),
                    decode_waiting: None,
                });
                tracing::debug!("Bootstrap: room {room_id} prefill arrived first");
                Some(rx)
            }
        };

        if let Some(rx) = rx {
            match tokio::time::timeout(RENDEZVOUS_TIMEOUT, rx).await {
                Ok(Ok(())) => {
                    tracing::debug!("Bootstrap: room {room_id} decode metadata received");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {room_id} decode metadata waiter dropped");
                }
                Err(_) => {
                    self.rooms.remove(&room_id);
                    bail!("Bootstrap: room {room_id} timeout waiting for decode metadata");
                }
            }
        }

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
                    decode_ready: false,
                    prefill_completed: true,
                    prefill_waiting: None,
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

/// Send decode receiver metadata to a prefill worker, then wait for KV to be ready.
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
    async fn test_prefill_waits_for_decode_metadata_before_completion() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 1004u64;

        let (prefill_entered_tx, prefill_entered_rx) = tokio::sync::oneshot::channel();
        let mut prefill_ready = tokio::spawn({
            let server = server.clone();
            async move {
                let _ = prefill_entered_tx.send(());
                server.wait_for_decode_ready(room_id).await
            }
        });

        prefill_entered_rx.await.unwrap();
        assert!(
            !prefill_ready.is_finished(),
            "Prefill should wait until decode metadata arrives"
        );

        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });

        let result = tokio::time::timeout(Duration::from_secs(1), &mut prefill_ready)
            .await
            .unwrap()
            .unwrap();
        assert!(
            result.is_ok(),
            "Prefill should see decode metadata: {result:?}"
        );

        assert!(
            !decode_handle.is_finished(),
            "Decode should wait until prefill marks the room complete"
        );

        server.complete_room(room_id);

        let result = tokio::time::timeout(Duration::from_secs(1), decode_handle)
            .await
            .unwrap()
            .unwrap();
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

    /// Two prefill tasks racing on `wait_for_decode_ready()` for the same room exercise the
    /// symmetric single-slot bug to the decode/decode race on `decode_waiting`: the second
    /// prefill's `(tx, rx)` overwrites the first's in `prefill_waiting`, causing the first
    /// prefill's receiver to get `Err(_)` ("sender dropped") and bail immediately.
    ///
    /// Explicit oneshot signals (not sleeps) enforce ordering: each task fires its signal
    /// before entering `wait_for_decode_ready`. Because `#[tokio::test]` uses the
    /// `current_thread` executor (cooperative, no preemption), by the time the test receives
    /// a signal the sending task has already run synchronously through the DashMap write
    /// inside `wait_for_decode_ready` and suspended on its internal `rx.await`. Decode is
    /// only spawned after both signals are in, so both prefills always encounter a room with
    /// `decode_ready=false`, making the overwrite deterministic and `assert_eq!(ok_count, 1)`
    /// exact rather than probabilistic.
    #[tokio::test]
    async fn test_two_concurrent_prefills_same_room() {
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 7001u64;

        // Each task signals just before calling wait_for_decode_ready. The cooperative
        // executor guarantees the DashMap write has completed before the test resumes:
        // there is no yield point between send() and the DashMap entry write, so the
        // task runs through both atomically before suspending on its internal rx.await.
        let (reg1_tx, reg1_rx) = oneshot::channel::<()>();
        let server1 = server.clone();
        let prefill1 = tokio::spawn(async move {
            let _ = reg1_tx.send(());
            server1.wait_for_decode_ready(room_id).await
        });

        let (reg2_tx, reg2_rx) = oneshot::channel::<()>();
        let server2 = server.clone();
        let prefill2 = tokio::spawn(async move {
            let _ = reg2_tx.send(());
            server2.wait_for_decode_ready(room_id).await
        });

        // Wait for both DashMap writes to complete before spawning decode.
        // Prefill1 always takes the Vacant path (creates the room); prefill2 always
        // finds it Occupied with decode_ready=false (decode has not yet connected)
        // and overwrites prefill_waiting, dropping prefill1's sender.
        reg1_rx.await.expect("prefill1 signal lost");
        reg2_rx.await.expect("prefill2 signal lost");

        // Now decode connects. handle_connection fires the surviving prefill_waiting
        // sender (prefill2's) and registers decode_waiting.
        let decode =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });

        // Unblock decode.
        server.complete_room(room_id);

        let r1 = tokio::time::timeout(Duration::from_secs(5), prefill1)
            .await
            .expect("prefill1 timed out")
            .expect("prefill1 panicked");
        let r2 = tokio::time::timeout(Duration::from_secs(5), prefill2)
            .await
            .expect("prefill2 timed out")
            .expect("prefill2 panicked");
        let rd = tokio::time::timeout(Duration::from_secs(5), decode)
            .await
            .expect("decode timed out")
            .expect("decode panicked");

        // Prefill1's sender was dropped by the overwrite → Err immediately.
        // Prefill2's sender survived → Ok when decode connected.
        // If prefill_waiting is ever upgraded from Option to Vec (supporting concurrent
        // waiters), both would return Ok and these assertions catch the change.
        let ok_count = [r1.is_ok(), r2.is_ok()].iter().filter(|&&ok| ok).count();
        assert_eq!(
            ok_count,
            1,
            "Exactly one prefill should succeed (surviving sender); r1={r1:?} r2={r2:?}"
        );
        let err_count = [r1.is_err(), r2.is_err()].iter().filter(|&&e| e).count();
        assert_eq!(
            err_count,
            1,
            "Exactly one prefill should fail (dropped sender); r1={r1:?} r2={r2:?}"
        );
        assert!(rd.is_ok(), "Decode should succeed: {rd:?}");

        cancel_token.cancel();
    }
}
