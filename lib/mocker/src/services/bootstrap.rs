// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap rendezvous for disaggregated mocker testing.
//!
//! Simulates the SGLang disaggregated serving handshake for KV transfer coordination.
//! Either prefill or decode can arrive first; prefill waits for decode metadata before
//! emitting output, and decode waits for prefill completion before generating.
//!
//! - Prefill: waits for decode metadata via `wait_for_decode_ready(room_id, timeout)`. When a
//!   DIS-2147 abort timeout is supplied and no decode arrives within it, prefill calls
//!   `abort_room(room_id)` so waiting/late decoders get a clean ABORT rather than hanging.
//! - Prefill: calls `complete_room(room_id)` after first token to release KV to decode (ACK).
//! - Decode: connects to prefill's bootstrap server, sends metadata, then waits for completion
//!   or abort. Decode is expected to connect only AFTER its own KV cache has capacity (DIS-2147).
//!
//! Wire protocol:
//! - Decode -> Prefill: room_id (8 bytes, little-endian u64)
//! - Prefill -> Decode: ACK (1 byte, 0x01) after prefill completes successfully
//! - Prefill -> Decode: ABORT (1 byte, 0x02) if prefill aborted before decode arrived

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

/// ACK byte sent from server to decode when prefill completes successfully.
const ACK_BYTE: u8 = 0x01;

/// ABORT byte sent from server to decode when prefill aborted before transfer (DIS-2147).
const ABORT_BYTE: u8 = 0x02;

/// How long an aborted room is retained so late-arriving decoders see ABORT instead of timing out.
const ABORTED_ROOM_TTL: Duration = Duration::from_secs(30);

/// Final outcome of a room's rendezvous.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoomOutcome {
    Pending,
    Completed,
    Aborted,
}

/// State for a room in the rendezvous.
struct RoomState {
    /// True if decode has sent receiver metadata for this room
    decode_ready: bool,
    /// Final outcome of this room (Pending until prefill completes or aborts).
    outcome: RoomOutcome,
    /// Channel to notify prefill when decode metadata arrives
    prefill_waiting: Option<oneshot::Sender<()>>,
    /// Channel to notify decode of the final outcome when prefill completes/aborts.
    decode_waiting: Option<oneshot::Sender<RoomOutcome>>,
}

impl RoomState {
    fn pending() -> Self {
        Self {
            decode_ready: false,
            outcome: RoomOutcome::Pending,
            prefill_waiting: None,
            decode_waiting: None,
        }
    }
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

    /// Handle a connection from decode. Marks decode ready (waking any waiting prefill), then
    /// blocks until prefill completes or aborts for this room.
    async fn handle_connection(
        mut stream: TcpStream,
        rooms: Arc<DashMap<u64, RoomState>>,
    ) -> Result<()> {
        // Read room_id (8 bytes, little-endian)
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf).await?;
        let room_id = u64::from_le_bytes(buf);

        tracing::debug!("Bootstrap: decode connected for room {room_id}");

        // Register decode metadata, wake prefill if it is waiting, then determine the response
        // byte immediately (if prefill already finished/aborted) or set up a wait.
        let immediate_or_wait: ImmediateOrWait = match rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().decode_ready = true;
                // If prefill is waiting for decode arrival, fire its signal now.
                if let Some(tx) = entry.get_mut().prefill_waiting.take() {
                    let _ = tx.send(());
                    tracing::debug!(
                        "Bootstrap: room {room_id} decode metadata unblocked waiting prefill"
                    );
                }
                match entry.get().outcome {
                    RoomOutcome::Completed => {
                        entry.remove();
                        tracing::debug!(
                            "Bootstrap: room {room_id} already completed, immediate ACK"
                        );
                        ImmediateOrWait::Immediate(ACK_BYTE)
                    }
                    RoomOutcome::Aborted => {
                        // Late decode arrives after prefill aborted — clean error (DIS-2147).
                        entry.remove();
                        tracing::warn!(
                            "Bootstrap: room {room_id} prefill aborted, sending ABORT to decode"
                        );
                        ImmediateOrWait::Immediate(ABORT_BYTE)
                    }
                    RoomOutcome::Pending => {
                        // Decode metadata is registered, but prefill has not completed yet. Wait.
                        let (tx, rx) = oneshot::channel();
                        entry.get_mut().decode_waiting = Some(tx);
                        tracing::debug!("Bootstrap: room {room_id} decode waiting for prefill");
                        ImmediateOrWait::Wait(rx)
                    }
                }
            }
            Entry::Vacant(entry) => {
                // Decode arrived first — create a Pending room, mark decode ready, and wait.
                let (tx, rx) = oneshot::channel();
                let mut state = RoomState::pending();
                state.decode_ready = true;
                state.decode_waiting = Some(tx);
                entry.insert(state);
                tracing::debug!("Bootstrap: room {room_id} decode arrived first, waiting");
                ImmediateOrWait::Wait(rx)
            }
        };

        // Wait for prefill if needed
        let response_byte = match immediate_or_wait {
            ImmediateOrWait::Immediate(b) => b,
            ImmediateOrWait::Wait(rx) => match tokio::time::timeout(RENDEZVOUS_TIMEOUT, rx).await {
                Ok(Ok(RoomOutcome::Completed)) => {
                    tracing::debug!("Bootstrap: room {room_id} prefill completed, sending ACK");
                    ACK_BYTE
                }
                Ok(Ok(RoomOutcome::Aborted)) => {
                    tracing::warn!(
                        "Bootstrap: room {room_id} prefill aborted while decode waited, \
                             sending ABORT"
                    );
                    ABORT_BYTE
                }
                Ok(Ok(RoomOutcome::Pending)) => {
                    bail!("Bootstrap: room {room_id} sender fired with Pending outcome");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {room_id} sender dropped");
                }
                Err(_) => {
                    rooms.remove(&room_id);
                    bail!("Bootstrap: room {room_id} timeout waiting for prefill");
                }
            },
        };

        stream.write_all(&[response_byte]).await?;
        Ok(())
    }

    /// Wait until decode has sent receiver metadata for this room. The act of decode connecting
    /// is its signal that it has KV capacity to receive the transfer (DIS-2147), so until then
    /// the caller (prefill) holds KV — modeling real NIXL backpressure.
    ///
    /// `abort_timeout` (DIS-2147): when `Some`, the wait is bounded by it and an Err is returned
    /// on timeout — the caller is expected to call [`abort_room`] so waiting/late decoders get a
    /// clean ABORT rather than hanging. When `None`, the wait is bounded only by
    /// [`RENDEZVOUS_TIMEOUT`] (pre-DIS-2147 behavior).
    pub async fn wait_for_decode_ready(
        &self,
        room_id: u64,
        abort_timeout: Option<Duration>,
    ) -> Result<()> {
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
                let mut state = RoomState::pending();
                state.prefill_waiting = Some(tx);
                entry.insert(state);
                tracing::debug!("Bootstrap: room {room_id} prefill arrived first");
                Some(rx)
            }
        };

        if let Some(rx) = rx {
            let wait = abort_timeout.unwrap_or(RENDEZVOUS_TIMEOUT);
            match tokio::time::timeout(wait, rx).await {
                Ok(Ok(())) => {
                    tracing::debug!("Bootstrap: room {room_id} decode metadata received");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {room_id} decode metadata waiter dropped");
                }
                Err(_) => {
                    // On abort-timeout, leave the room in place so abort_room can mark it Aborted
                    // for waiting/late decodes. Otherwise (pre-DIS-2147) remove it.
                    if abort_timeout.is_none() {
                        self.rooms.remove(&room_id);
                    }
                    bail!("Bootstrap: room {room_id} timeout waiting for decode metadata");
                }
            }
        }

        Ok(())
    }

    /// Mark a room as completed (prefill finished, KV cache ready). If decode is already waiting,
    /// unblocks it with ACK.
    pub fn complete_room(&self, room_id: u64) {
        self.set_outcome(room_id, RoomOutcome::Completed);
    }

    /// Mark a room as aborted (prefill timed out waiting for decode, or other failure). Any
    /// already-waiting decode receives ABORT. The room is retained for [`ABORTED_ROOM_TTL`] so
    /// late-arriving decodes also see ABORT rather than hanging until RENDEZVOUS_TIMEOUT. (DIS-2147)
    pub fn abort_room(&self, room_id: u64) {
        self.set_outcome(room_id, RoomOutcome::Aborted);
        // Schedule cleanup so the room doesn't leak forever after a late decode also fails to show
        let rooms = self.rooms.clone();
        tokio::spawn(async move {
            tokio::time::sleep(ABORTED_ROOM_TTL).await;
            // Only remove if still in Aborted state (a late decode may have already removed it)
            if let Entry::Occupied(entry) = rooms.entry(room_id)
                && entry.get().outcome == RoomOutcome::Aborted
            {
                entry.remove();
                tracing::debug!("Bootstrap: aborted room {room_id} TTL expired, cleaned up");
            }
        });
    }

    fn set_outcome(&self, room_id: u64, outcome: RoomOutcome) {
        match self.rooms.entry(room_id) {
            Entry::Occupied(mut entry) => {
                let state = entry.get_mut();
                state.outcome = outcome;
                // Fire any waiting decode with the outcome
                if let Some(tx) = state.decode_waiting.take() {
                    let _ = tx.send(outcome);
                    if outcome == RoomOutcome::Completed {
                        // Successful handoff — room no longer needed
                        entry.remove();
                    }
                    // If Aborted, room is retained for the TTL window so other late decodes
                    // (this one was already here) also see ABORT.
                }
                // If no decode_waiting, the outcome is now persistent on the entry; a late decode
                // will read it directly in handle_connection.
            }
            Entry::Vacant(entry) => {
                let mut state = RoomState::pending();
                state.outcome = outcome;
                entry.insert(state);
                tracing::debug!(
                    "Bootstrap: room {room_id} outcome set to {outcome:?} (no decode yet)"
                );
            }
        }
    }

    /// Get the port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Internal helper enum for the handle_connection decision.
enum ImmediateOrWait {
    Immediate(u8),
    Wait(oneshot::Receiver<RoomOutcome>),
}

/// Send decode receiver metadata to a prefill worker, then wait for KV to be ready.
/// Returns Err on ABORT_BYTE (prefill timed out before transfer — DIS-2147).
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

    // Wait for response byte (blocks until prefill completes or aborts)
    let mut response = [0u8; 1];
    tokio::time::timeout(RENDEZVOUS_TIMEOUT, stream.read_exact(&mut response))
        .await
        .map_err(|_| anyhow::anyhow!("Bootstrap: response timeout for room {room_id}"))?
        .map_err(|e| anyhow::anyhow!("Bootstrap: read response failed: {e}"))?;

    match response[0] {
        ACK_BYTE => {
            tracing::debug!("Bootstrap: decode received ACK for room {room_id}");
            Ok(())
        }
        ABORT_BYTE => {
            tracing::warn!("Bootstrap: prefill aborted transfer for room {room_id}");
            bail!("Bootstrap: prefill aborted before transfer (room {room_id})");
        }
        other => bail!(
            "Bootstrap: invalid response byte {:02x} for room {room_id}",
            other
        ),
    }
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
                server.wait_for_decode_ready(room_id, None).await
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

    // DIS-2147 — new scenario tests

    #[tokio::test]
    async fn test_wait_for_decode_arrival_decode_present_first() {
        // Decode arrives first; prefill's subsequent wait returns immediately.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 3001u64;

        // Decode connects first (registers as decode_waiting)
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Prefill calls wait_for_decode_arrival -> should return immediately
        let wait_start = std::time::Instant::now();
        let wait_result = server
            .wait_for_decode_ready(room_id, Some(Duration::from_secs(5)))
            .await;
        let wait_elapsed = wait_start.elapsed();
        assert!(wait_result.is_ok(), "wait should succeed: {wait_result:?}");
        assert!(
            wait_elapsed < Duration::from_millis(50),
            "wait should return immediately, took {wait_elapsed:?}"
        );

        // Prefill now completes
        server.complete_room(room_id);

        let decode_result = decode_handle.await.unwrap();
        assert!(
            decode_result.is_ok(),
            "decode should succeed: {decode_result:?}"
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_wait_for_decode_arrival_prefill_waits_then_decode_arrives() {
        // Prefill waits first; decode arrives during the wait; prefill's wait unblocks.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 3002u64;

        // Prefill starts waiting (room doesn't exist yet)
        let server_clone = server.clone();
        let wait_handle = tokio::spawn(async move {
            server_clone
                .wait_for_decode_ready(room_id, Some(Duration::from_secs(5)))
                .await
        });

        // Decode arrives shortly after
        tokio::time::sleep(Duration::from_millis(50)).await;
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });

        // Prefill's wait should unblock
        let wait_result = wait_handle.await.unwrap();
        assert!(
            wait_result.is_ok(),
            "wait_for_decode_arrival should succeed: {wait_result:?}"
        );

        // Prefill completes; decode gets ACK
        server.complete_room(room_id);
        let decode_result = decode_handle.await.unwrap();
        assert!(
            decode_result.is_ok(),
            "decode should succeed: {decode_result:?}"
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_wait_for_decode_arrival_timeout_then_abort() {
        // Prefill waits, no decode arrives, prefill times out and aborts the room.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let room_id = 3003u64;

        // Prefill waits with a short timeout; no decode shows up
        let wait_result = server
            .wait_for_decode_ready(room_id, Some(Duration::from_millis(100)))
            .await;
        assert!(wait_result.is_err(), "wait should time out");

        // Prefill marks the room aborted
        server.abort_room(room_id);

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_late_decode_on_aborted_room_gets_abort_byte() {
        // Prefill aborts a room; a decode arriving afterwards within the TTL window receives ABORT.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 3004u64;

        // Prefill marks the room aborted directly (simulating: it had no decode arrive in time)
        server.abort_room(room_id);

        // Decode now connects — should receive ABORT_BYTE and return a clean error
        let decode_result = connect_to_prefill("127.0.0.1", port, room_id).await;
        assert!(
            decode_result.is_err(),
            "decode should receive abort, got: {decode_result:?}"
        );
        let err_msg = format!("{:#}", decode_result.unwrap_err());
        assert!(
            err_msg.contains("aborted"),
            "error should mention aborted, got: {err_msg}"
        );

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn test_decode_waiting_gets_abort_when_prefill_aborts() {
        // Decode connects first and is waiting; prefill aborts; the waiting decode receives ABORT.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let room_id = 3005u64;

        // Decode connects first and waits
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, room_id).await });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Prefill aborts (simulating decode-arrival timeout from prefill's perspective)
        server.abort_room(room_id);

        // Decode should error with abort
        let decode_result = decode_handle.await.unwrap();
        assert!(
            decode_result.is_err(),
            "decode should receive abort, got: {decode_result:?}"
        );
        let err_msg = format!("{:#}", decode_result.unwrap_err());
        assert!(
            err_msg.contains("aborted"),
            "error should mention aborted, got: {err_msg}"
        );

        cancel_token.cancel();
    }
}
