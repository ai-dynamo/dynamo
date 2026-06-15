// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bootstrap rendezvous for disaggregated mocker testing.
//!
//! Simulates the disaggregated serving handshake for KV transfer coordination.
//! Either prefill or decode can arrive first; prefill waits for decode metadata before
//! emitting output, and decode waits for prefill completion before generating.
//!
//! ## Channel key: `transfer_id`
//!
//! The channel is keyed by a single engine-neutral `u64` named `transfer_id`: it
//! correlates one prefill→decode KV transfer regardless of engine. The two
//! engine vocabularies are the SAME `u64` value — there is no conversion:
//!
//! - **sglang** computes a *bootstrap room* up front (frontend
//!   `compute_bootstrap_room`) and passes it as a [`RoomId`].
//! - **vLLM** emits a *transfer id* in its prefill output
//!   (`disaggregated_params.transfer_id`) and passes it as a [`TransferId`].
//!
//! Both alias to `u64` and hit the same map entry, so the connect/wait/ACK/ABORT
//! machinery below is identical for both.
//!
//! - Prefill: waits for decode metadata via `wait_for_decode_ready(transfer_id, timeout)`. When a
//!   abort timeout is supplied and no decode arrives within it, prefill calls
//!   `abort_room(transfer_id)` so waiting/late decoders get a clean ABORT rather than hanging.
//! - Prefill: calls `complete_room(transfer_id)` after first token to release KV to decode (ACK).
//! - Decode: connects to prefill's bootstrap server, sends metadata, then waits for completion
//!   or abort. Decode is expected to connect only AFTER its own KV cache has capacity.
//!
//! Wire protocol:
//! - Decode -> Prefill: transfer_id (8 bytes, little-endian u64)
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

/// sglang vocabulary for the channel key (bootstrap triple). Same `u64` as
/// [`TransferId`]; the two are interchangeable and hit the same map entry.
pub type RoomId = u64;
/// vLLM vocabulary for the channel key (`kv_transfer_params`). Same `u64` as
/// [`RoomId`]; the two are interchangeable and hit the same map entry.
pub type TransferId = u64;

/// Timeout for bootstrap rendezvous operations.
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// ACK byte sent from server to decode when prefill completes successfully.
const ACK_BYTE: u8 = 0x01;

/// ABORT byte sent from server to decode when prefill aborted before transfer.
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

/// A registered prefill-side KV pin awaiting its decode pull. The prefill
/// process owns the long-lived channel server; at pin time it registers the
/// release trigger here, keyed by `transfer_id`. The trigger fires exactly
/// once — when the matching decode connects (release the pin, then ACK), or
/// when `abort_timeout` elapses with no decode (release the pin + ABORT). This
/// is the NIXL-faithful model: the connector holds the KV independent of the
/// prefill *request* lifecycle (which has already streamed-complete) until the
/// decode pulls.
struct PinRegistration {
    /// Fires the scheduler's `release_pin` for the pinned prefill `uuid`.
    /// Boxed so the channel layer needs no knowledge of the scheduler types.
    /// `Option` so it can be taken and fired at-most-once.
    release: Option<Box<dyn FnOnce() + Send + Sync>>,
    /// Cancels the abort-timeout timer task. On the happy path the decode
    /// connects in ~ms and releases the pin; cancelling stops the (otherwise
    /// up-to-30s) timer task from lingering, so parked tasks track live
    /// transfers rather than traffic × timeout.
    abort_timer: Option<tokio::task::AbortHandle>,
}

/// Bootstrap server for prefill mockers.
/// Handles rendezvous between prefill and decode for KV transfer coordination.
pub struct BootstrapServer {
    port: u16,
    rooms: Arc<DashMap<u64, RoomState>>,
    /// Prefill-side pin registry, keyed by `transfer_id`. The release trigger is
    /// driven HERE (decode connect / abort-timeout) — decoupled from the prefill
    /// request stream, which completes normally and does NOT block on the pull.
    pins: Arc<DashMap<u64, PinRegistration>>,
}

impl BootstrapServer {
    /// Start the bootstrap server on the specified port.
    pub async fn start(port: u16, cancel_token: CancellationToken) -> Result<Arc<Self>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
        let actual_port = listener.local_addr()?.port();

        tracing::info!("Bootstrap server started on port {actual_port}");

        let rooms: Arc<DashMap<u64, RoomState>> = Arc::new(DashMap::new());
        let pins: Arc<DashMap<u64, PinRegistration>> = Arc::new(DashMap::new());
        let server = Arc::new(Self {
            port: actual_port,
            rooms: rooms.clone(),
            pins: pins.clone(),
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
                                let pins_clone = pins.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_connection(stream, rooms_clone, pins_clone).await {
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
        pins: Arc<DashMap<u64, PinRegistration>>,
    ) -> Result<()> {
        // Read transfer_id (8 bytes, little-endian)
        let mut buf = [0u8; 8];
        stream.read_exact(&mut buf).await?;
        let transfer_id = u64::from_le_bytes(buf);

        tracing::debug!("Bootstrap: decode connected for room {transfer_id}");

        // The decode has connected to pull this transfer: release the prefill's
        // pinned KV NOW (the strand ends as the decode pulls). This is the live
        // release trigger — decoupled from the prefill request stream, which has
        // already completed. Fires the registered scheduler `release_pin`
        // at-most-once; a decode for a transfer with no pin (already released, or
        // aggregated) is a harmless no-op.
        Self::fire_pin_release(&pins, transfer_id);

        // Register decode metadata, wake prefill if it is waiting, then determine the response
        // byte immediately (if prefill already finished/aborted) or set up a wait.
        let immediate_or_wait: ImmediateOrWait = match rooms.entry(transfer_id) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().decode_ready = true;
                // If prefill is waiting for decode arrival, fire its signal now.
                if let Some(tx) = entry.get_mut().prefill_waiting.take() {
                    let _ = tx.send(());
                    tracing::debug!(
                        "Bootstrap: room {transfer_id} decode metadata unblocked waiting prefill"
                    );
                }
                match entry.get().outcome {
                    RoomOutcome::Completed => {
                        entry.remove();
                        tracing::debug!(
                            "Bootstrap: room {transfer_id} already completed, immediate ACK"
                        );
                        ImmediateOrWait::Immediate(ACK_BYTE)
                    }
                    RoomOutcome::Aborted => {
                        // Late decode arrives after prefill aborted — clean error.
                        entry.remove();
                        tracing::warn!(
                            "Bootstrap: room {transfer_id} prefill aborted, sending ABORT to decode"
                        );
                        ImmediateOrWait::Immediate(ABORT_BYTE)
                    }
                    RoomOutcome::Pending => {
                        // Decode metadata is registered, but prefill has not completed yet. Wait.
                        let (tx, rx) = oneshot::channel();
                        entry.get_mut().decode_waiting = Some(tx);
                        tracing::debug!("Bootstrap: room {transfer_id} decode waiting for prefill");
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
                tracing::debug!("Bootstrap: room {transfer_id} decode arrived first, waiting");
                ImmediateOrWait::Wait(rx)
            }
        };

        // Wait for prefill if needed
        let response_byte = match immediate_or_wait {
            ImmediateOrWait::Immediate(b) => b,
            ImmediateOrWait::Wait(rx) => match tokio::time::timeout(RENDEZVOUS_TIMEOUT, rx).await {
                Ok(Ok(RoomOutcome::Completed)) => {
                    tracing::debug!("Bootstrap: room {transfer_id} prefill completed, sending ACK");
                    ACK_BYTE
                }
                Ok(Ok(RoomOutcome::Aborted)) => {
                    tracing::warn!(
                        "Bootstrap: room {transfer_id} prefill aborted while decode waited, \
                             sending ABORT"
                    );
                    ABORT_BYTE
                }
                Ok(Ok(RoomOutcome::Pending)) => {
                    bail!("Bootstrap: room {transfer_id} sender fired with Pending outcome");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {transfer_id} sender dropped");
                }
                Err(_) => {
                    rooms.remove(&transfer_id);
                    bail!("Bootstrap: room {transfer_id} timeout waiting for prefill");
                }
            },
        };

        stream.write_all(&[response_byte]).await?;
        Ok(())
    }

    /// Wait until decode has sent receiver metadata for this room. The act of decode connecting
    /// is its signal that it has KV capacity to receive the transfer, so until then
    /// the caller (prefill) holds KV — modeling real NIXL backpressure.
    ///
    /// `abort_timeout`: when `Some`, the wait is bounded by it and an Err is returned
    /// on timeout — the caller is expected to call [`abort_room`] so waiting/late decoders get a
    /// clean ABORT rather than hanging. When `None`, the wait is bounded only by
    /// [`RENDEZVOUS_TIMEOUT`] (legacy behavior).
    pub async fn wait_for_decode_ready(
        &self,
        transfer_id: u64,
        abort_timeout: Option<Duration>,
    ) -> Result<()> {
        let rx = match self.rooms.entry(transfer_id) {
            Entry::Occupied(mut entry) => {
                if entry.get().decode_ready {
                    tracing::debug!("Bootstrap: room {transfer_id} decode already ready");
                    None
                } else {
                    let (tx, rx) = oneshot::channel();
                    entry.get_mut().prefill_waiting = Some(tx);
                    tracing::debug!(
                        "Bootstrap: room {transfer_id} prefill waiting for decode metadata"
                    );
                    Some(rx)
                }
            }
            Entry::Vacant(entry) => {
                let (tx, rx) = oneshot::channel();
                let mut state = RoomState::pending();
                state.prefill_waiting = Some(tx);
                entry.insert(state);
                tracing::debug!("Bootstrap: room {transfer_id} prefill arrived first");
                Some(rx)
            }
        };

        if let Some(rx) = rx {
            let wait = abort_timeout.unwrap_or(RENDEZVOUS_TIMEOUT);
            match tokio::time::timeout(wait, rx).await {
                Ok(Ok(())) => {
                    tracing::debug!("Bootstrap: room {transfer_id} decode metadata received");
                }
                Ok(Err(_)) => {
                    bail!("Bootstrap: room {transfer_id} decode metadata waiter dropped");
                }
                Err(_) => {
                    // On abort-timeout, leave the room in place so abort_room can mark it Aborted
                    // for waiting/late decodes. Otherwise (legacy) remove it.
                    if abort_timeout.is_none() {
                        self.rooms.remove(&transfer_id);
                    }
                    bail!("Bootstrap: room {transfer_id} timeout waiting for decode metadata");
                }
            }
        }

        Ok(())
    }

    /// Mark a room as completed (prefill finished, KV cache ready). If decode is already waiting,
    /// unblocks it with ACK.
    pub fn complete_room(&self, transfer_id: u64) {
        self.set_outcome(transfer_id, RoomOutcome::Completed);
    }

    /// Mark a room as aborted (prefill timed out waiting for decode, or other failure). Any
    /// already-waiting decode receives ABORT. The room is retained for [`ABORTED_ROOM_TTL`] so
    /// late-arriving decodes also see ABORT rather than hanging until RENDEZVOUS_TIMEOUT.
    pub fn abort_room(&self, transfer_id: u64) {
        self.set_outcome(transfer_id, RoomOutcome::Aborted);
        // Schedule cleanup so the room doesn't leak forever after a late decode also fails to show
        let rooms = self.rooms.clone();
        tokio::spawn(async move {
            tokio::time::sleep(ABORTED_ROOM_TTL).await;
            // Only remove if still in Aborted state (a late decode may have already removed it)
            if let Entry::Occupied(entry) = rooms.entry(transfer_id)
                && entry.get().outcome == RoomOutcome::Aborted
            {
                entry.remove();
                tracing::debug!("Bootstrap: aborted room {transfer_id} TTL expired, cleaned up");
            }
        });
    }

    fn set_outcome(&self, transfer_id: u64, outcome: RoomOutcome) {
        match self.rooms.entry(transfer_id) {
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
                    "Bootstrap: room {transfer_id} outcome set to {outcome:?} (no decode yet)"
                );
            }
        }
    }

    /// Register a prefill-side KV pin awaiting its decode pull, keyed by
    /// `transfer_id`. The prefill calls this at pin time (right after its KV is
    /// pinned on prefill-complete) and then lets its request stream COMPLETE
    /// normally — it does NOT block on the pull. The pin is released by the
    /// channel server when:
    /// - the matching decode connects ([`fire_pin_release`] from
    ///   `handle_connection`) → release pin, then ACK the decode; or
    /// - `abort_timeout` (when `Some`) elapses with no decode → release pin +
    ///   mark the room Aborted so a late decode gets a clean ABORT.
    ///
    /// `release` fires the scheduler's `release_pin` for the pinned prefill; it
    /// is boxed so the channel layer carries no scheduler types and is invoked
    /// at-most-once. The room is marked Completed so a connecting decode pulls
    /// immediately (the prefill's KV is ready the moment it is pinned).
    pub fn register_pin(
        self: &Arc<Self>,
        transfer_id: u64,
        abort_timeout: Option<Duration>,
        release: Box<dyn FnOnce() + Send + Sync>,
    ) {
        self.pins.insert(
            transfer_id,
            PinRegistration {
                release: Some(release),
                abort_timer: None,
            },
        );
        // The prefill's KV is pinned and ready to pull; mark the room complete so
        // a connecting decode receives an immediate ACK.
        self.complete_room(transfer_id);

        // Bound the strand so a pinned KV is ALWAYS reclaimable: an explicit
        // `abort_timeout` when set, else `RENDEZVOUS_TIMEOUT` — the same fallback
        // sglang's `wait_for_decode_ready` already applies (parity). Without this
        // bound a decode no-show / transient connect failure would leak the pin
        // forever (its blocks stay counted active, draining the pool). No decode
        // by the deadline → release the pin and ABORT so waiting/late decodes get
        // a clean abort rather than hanging.
        let timeout = abort_timeout.unwrap_or(RENDEZVOUS_TIMEOUT);
        let server = self.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            // Still pending → no decode pulled within the abort window.
            if Self::fire_pin_release(&server.pins, transfer_id) {
                tracing::warn!(
                    "Bootstrap: transfer {transfer_id} abort-timeout — releasing pin, aborting"
                );
                server.abort_room(transfer_id);
            }
        });
        // Record the timer's abort handle so the decode-connect release can cancel
        // it (happy path: decode pulls in ~ms, no 30s task lingers). If the decode
        // already released in the tiny window above, the entry is gone and the
        // timer simply no-ops when it fires.
        if let Some(mut reg) = self.pins.get_mut(&transfer_id) {
            reg.abort_timer = Some(handle.abort_handle());
        }
    }

    /// Fire (at-most-once) the release trigger for a registered pin and remove
    /// the registration. Returns `true` if a pin was present and released — used
    /// by the abort timer to distinguish "decode already pulled" (false) from
    /// "no decode showed up" (true). Idempotent / safe on an unknown transfer.
    fn fire_pin_release(pins: &DashMap<u64, PinRegistration>, transfer_id: u64) -> bool {
        if let Some((_, mut reg)) = pins.remove(&transfer_id)
            && let Some(release) = reg.release.take()
        {
            release();
            // Decode pulled (or we are the timer firing): cancel the abort timer
            // so it does not linger to its deadline. Aborting the currently-firing
            // timer task is a harmless no-op.
            if let Some(timer) = reg.abort_timer.take() {
                timer.abort();
            }
            return true;
        }
        false
    }

    /// Test/forensic accessor: number of prefill pins currently registered
    /// (awaiting their decode pull).
    #[cfg(test)]
    pub fn num_registered_pins(&self) -> usize {
        self.pins.len()
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
/// Returns Err on ABORT_BYTE (prefill timed out before transfer).
pub async fn connect_to_prefill(host: &str, port: u16, transfer_id: u64) -> Result<()> {
    let host = host.trim_matches(|c| c == '[' || c == ']');
    let addr = format!("{host}:{port}");

    tracing::debug!("Bootstrap: decode connecting to {addr} for room {transfer_id}");

    // Connect with timeout
    let mut stream = tokio::time::timeout(RENDEZVOUS_TIMEOUT, TcpStream::connect(&addr))
        .await
        .map_err(|_| anyhow::anyhow!("Bootstrap: connect timeout to {addr}"))?
        .map_err(|e| anyhow::anyhow!("Bootstrap: connect failed to {addr}: {e}"))?;

    // Send transfer_id
    stream.write_all(&transfer_id.to_le_bytes()).await?;

    // Wait for response byte (blocks until prefill completes or aborts)
    let mut response = [0u8; 1];
    tokio::time::timeout(RENDEZVOUS_TIMEOUT, stream.read_exact(&mut response))
        .await
        .map_err(|_| anyhow::anyhow!("Bootstrap: response timeout for room {transfer_id}"))?
        .map_err(|e| anyhow::anyhow!("Bootstrap: read response failed: {e}"))?;

    match response[0] {
        ACK_BYTE => {
            tracing::debug!("Bootstrap: decode received ACK for room {transfer_id}");
            Ok(())
        }
        ABORT_BYTE => {
            tracing::warn!("Bootstrap: prefill aborted transfer for room {transfer_id}");
            bail!("Bootstrap: prefill aborted before transfer (room {transfer_id})");
        }
        other => bail!(
            "Bootstrap: invalid response byte {:02x} for room {transfer_id}",
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
        let transfer_id = 1001u64;

        // Prefill completes first
        server.complete_room(transfer_id);

        // Decode connects - should get immediate ACK
        let result = connect_to_prefill("127.0.0.1", port, transfer_id).await;
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
        let transfer_id = 1002u64;

        // Spawn decode (will block waiting for prefill)
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, transfer_id).await });

        // Give decode time to connect and register
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Prefill completes - should unblock decode
        server.complete_room(transfer_id);

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
        let transfer_id = 1004u64;

        let (prefill_entered_tx, prefill_entered_rx) = tokio::sync::oneshot::channel();
        let mut prefill_ready = tokio::spawn({
            let server = server.clone();
            async move {
                let _ = prefill_entered_tx.send(());
                server.wait_for_decode_ready(transfer_id, None).await
            }
        });

        prefill_entered_rx.await.unwrap();
        assert!(
            !prefill_ready.is_finished(),
            "Prefill should wait until decode metadata arrives"
        );

        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, transfer_id).await });

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

        server.complete_room(transfer_id);

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
        let transfer_id = 1003u64;

        // Spawn decode
        let server_clone = server.clone();
        let decode_handle = tokio::spawn(async move {
            // Small delay so prefill can "register" conceptually first
            tokio::time::sleep(Duration::from_millis(10)).await;
            connect_to_prefill("127.0.0.1", port, transfer_id).await
        });

        // Prefill completes after decode starts connecting
        tokio::time::sleep(Duration::from_millis(50)).await;
        server_clone.complete_room(transfer_id);

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
        let transfer_id = 9999u64;

        // Decode connects but prefill never completes - use short timeout
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            connect_to_prefill("127.0.0.1", port, transfer_id),
        )
        .await;

        // Should timeout (outer timeout, not inner RENDEZVOUS_TIMEOUT)
        assert!(result.is_err(), "Should timeout waiting for prefill");

        cancel_token.cancel();
    }

    // Abort-timeout scenario tests

    #[tokio::test]
    async fn test_wait_for_decode_arrival_decode_present_first() {
        // Decode arrives first; prefill's subsequent wait returns immediately.
        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();

        let port = server.port();
        let transfer_id = 3001u64;

        // Decode connects first (registers as decode_waiting)
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, transfer_id).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Prefill calls wait_for_decode_arrival -> should return immediately
        let wait_start = std::time::Instant::now();
        let wait_result = server
            .wait_for_decode_ready(transfer_id, Some(Duration::from_secs(5)))
            .await;
        let wait_elapsed = wait_start.elapsed();
        assert!(wait_result.is_ok(), "wait should succeed: {wait_result:?}");
        assert!(
            wait_elapsed < Duration::from_millis(50),
            "wait should return immediately, took {wait_elapsed:?}"
        );

        // Prefill now completes
        server.complete_room(transfer_id);

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
        let transfer_id = 3002u64;

        // Prefill starts waiting (room doesn't exist yet)
        let server_clone = server.clone();
        let wait_handle = tokio::spawn(async move {
            server_clone
                .wait_for_decode_ready(transfer_id, Some(Duration::from_secs(5)))
                .await
        });

        // Decode arrives shortly after
        tokio::time::sleep(Duration::from_millis(50)).await;
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, transfer_id).await });

        // Prefill's wait should unblock
        let wait_result = wait_handle.await.unwrap();
        assert!(
            wait_result.is_ok(),
            "wait_for_decode_arrival should succeed: {wait_result:?}"
        );

        // Prefill completes; decode gets ACK
        server.complete_room(transfer_id);
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

        let transfer_id = 3003u64;

        // Prefill waits with a short timeout; no decode shows up
        let wait_result = server
            .wait_for_decode_ready(transfer_id, Some(Duration::from_millis(100)))
            .await;
        assert!(wait_result.is_err(), "wait should time out");

        // Prefill marks the room aborted
        server.abort_room(transfer_id);

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
        let transfer_id = 3004u64;

        // Prefill marks the room aborted directly (simulating: it had no decode arrive in time)
        server.abort_room(transfer_id);

        // Decode now connects — should receive ABORT_BYTE and return a clean error
        let decode_result = connect_to_prefill("127.0.0.1", port, transfer_id).await;
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

    /// Channel-server-driven release (the live vLLM model): the prefill
    /// registers a pin and lets its request stream complete; the pin persists
    /// until a decode connects for that `transfer_id`, at which point the server
    /// fires the release (here, sets a flag) and ACKs the decode. The prefill
    /// never blocks on the pull.
    #[tokio::test]
    async fn test_decode_connect_fires_pin_release() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();
        let port = server.port();
        let transfer_id = 7001u64;

        // Prefill pins and registers the release; its stream is now free to end.
        let released = Arc::new(AtomicBool::new(false));
        let flag = released.clone();
        server.register_pin(
            transfer_id,
            Some(Duration::from_secs(5)),
            Box::new(move || flag.store(true, Ordering::SeqCst)),
        );
        assert_eq!(
            server.num_registered_pins(),
            1,
            "pin persists after register"
        );
        assert!(
            !released.load(Ordering::SeqCst),
            "pin not released until a decode pulls"
        );

        // Decode connects → server fires release, then ACKs.
        let result = connect_to_prefill("127.0.0.1", port, transfer_id).await;
        assert!(result.is_ok(), "decode should get ACK: {result:?}");
        assert!(
            released.load(Ordering::SeqCst),
            "decode connect must release the pin"
        );
        assert_eq!(
            server.num_registered_pins(),
            0,
            "registration consumed on release"
        );

        cancel_token.cancel();
    }

    /// Abort-timeout with no decode: the server releases the pin and aborts the
    /// room, so a late decode receives a clean ABORT. The release fires exactly
    /// once (the abort timer path, since no decode showed up).
    #[tokio::test]
    async fn test_pin_abort_timeout_releases_and_aborts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cancel_token = CancellationToken::new();
        let server = BootstrapServer::start(0, cancel_token.clone())
            .await
            .unwrap();
        let port = server.port();
        let transfer_id = 7002u64;

        let releases = Arc::new(AtomicUsize::new(0));
        let counter = releases.clone();
        server.register_pin(
            transfer_id,
            Some(Duration::from_millis(50)),
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );

        // Let the abort timer fire (no decode shows up).
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            releases.load(Ordering::SeqCst),
            1,
            "abort-timeout releases the pin exactly once"
        );

        // A late decode now connects → must get a clean ABORT, and the release
        // does NOT fire a second time.
        let decode_result = connect_to_prefill("127.0.0.1", port, transfer_id).await;
        assert!(
            decode_result.is_err(),
            "late decode should receive ABORT: {decode_result:?}"
        );
        assert_eq!(
            releases.load(Ordering::SeqCst),
            1,
            "release is at-most-once (no double release)"
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
        let transfer_id = 3005u64;

        // Decode connects first and waits
        let decode_handle =
            tokio::spawn(async move { connect_to_prefill("127.0.0.1", port, transfer_id).await });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Prefill aborts (simulating decode-arrival timeout from prefill's perspective)
        server.abort_room(transfer_id);

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
