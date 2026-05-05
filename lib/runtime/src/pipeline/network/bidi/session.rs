// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared session machinery used by both client and server in a bidi exchange.
//!
//! - [`BidiSession`] bundles a [`Controller`] (so the user-visible `Context`
//!   stays in scope across the two pumps) with a `peer_done_observed` notify
//!   that the closer state machine waits on before finalizing.
//! - [`BidiSession::run_outgoing`] is the upstream/downstream pump used
//!   identically on both sides: drain the source `Stream<T>`, ship `Data` /
//!   `Done` frames, wait for peer-done, finalize.
//! - [`decode_bidi_anchor`] is the consumer-side adapter: turns a velo
//!   `StreamAnchor<BidiFrame<T>>` into a flat `DataStream<T>`, swallowing
//!   `Done` (it just notifies the closer) and `Finalized`.
//! - [`SessionGuard`] is a tiny RAII helper that calls `controller.kill()` on
//!   drop — embed it in any struct that owns the user-visible response stream
//!   so dropping the stream tears the session down.

use std::sync::Arc;

use futures::StreamExt;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::sync::Notify;
use tracing::{debug, trace, warn};

use velo::streaming::{StreamAnchor, StreamError, StreamFrame, StreamSender};

use crate::engine::{AsyncEngineContext, DataStream};
use crate::pipeline::context::Controller;
use crate::pipeline::network::bidi::BidiFrame;

/// Outcome of the outgoing pump, kept so callers can record metrics or react
/// to abnormal closes (logging, surfacing errors). Not surfaced to user code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionOutcome {
    /// Source exhausted, peer also signaled `Done`, sender finalized cleanly.
    Finalized,
    /// Either the session controller was killed, the peer cancelled our
    /// sender, or a send failed before we could finalize. Sender was dropped
    /// without `finalize`, so the peer will observe `Dropped`.
    Cancelled,
}

/// Shared per-session state. One instance per logical bidi session, on each
/// side. Wrapped in `Arc` so the pump task and the consumer wrapper can share
/// access to `peer_done_observed` and `controller`.
pub struct BidiSession {
    controller: Arc<Controller>,
    peer_done_observed: Arc<Notify>,
}

impl BidiSession {
    pub fn new(request_id: String) -> Arc<Self> {
        Arc::new(Self {
            controller: Arc::new(Controller::new(request_id)),
            peer_done_observed: Arc::new(Notify::new()),
        })
    }

    /// The controller used as the session's `Context` controller. Both pumps
    /// observe `controller.killed()` for early abort, and any external party
    /// (e.g. [`SessionGuard`] on drop, or a user explicit cancel) can call
    /// `controller.kill()` to tear the session down.
    pub fn controller(&self) -> Arc<Controller> {
        self.controller.clone()
    }

    /// Trait-object view of the controller, suitable for `ResponseStream::new`.
    pub fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.controller.clone()
    }

    /// Notify that fires the first time our consumer-side anchor reads a
    /// `BidiFrame::Done` from the peer. The pump task waits on this before
    /// calling `finalize` (per the closer state machine: hold the velo stream
    /// open after sending our own Done so future Control responses can flow).
    pub fn peer_done(&self) -> Arc<Notify> {
        self.peer_done_observed.clone()
    }

    /// Drive the outgoing direction: pump items from `source` through
    /// `outgoing` as `BidiFrame::Data(t)`. When `source` ends, send
    /// `BidiFrame::Done`, wait until our consumer side observes the peer's
    /// `Done`, then `finalize`.
    ///
    /// Aborts (drops `outgoing` so the peer sees `Dropped`) when:
    /// - `controller.killed()` fires (e.g. user dropped the response, server
    ///   handler errored, peer cancelled us).
    /// - `outgoing.cancellation_token()` fires (peer cancelled our anchor).
    /// - Any `outgoing.send` fails (peer's anchor is gone).
    pub async fn run_outgoing<T>(
        self: &Arc<Self>,
        mut source: DataStream<T>,
        outgoing: StreamSender<BidiFrame<T>>,
    ) -> SessionOutcome
    where
        T: Serialize + Send + 'static,
    {
        let cancel = outgoing.cancellation_token();

        // Phase 1: drain source, emitting Data frames. Exit on cancellation
        // signals, or break out when source ends so we can send Done.
        loop {
            tokio::select! {
                biased;
                _ = self.controller.killed() => {
                    debug!(id = %self.controller.id(), "outgoing pump: controller killed");
                    drop(outgoing);
                    return SessionOutcome::Cancelled;
                }
                _ = cancel.cancelled() => {
                    debug!(id = %self.controller.id(), "outgoing pump: peer cancelled our sender");
                    drop(outgoing);
                    return SessionOutcome::Cancelled;
                }
                item = source.next() => match item {
                    Some(t) => {
                        if let Err(e) = outgoing.send(BidiFrame::Data(t)).await {
                            warn!(id = %self.controller.id(), error = %e, "outgoing pump: send Data failed");
                            // Drop without finalize -> Dropped sentinel to peer.
                            return SessionOutcome::Cancelled;
                        }
                    }
                    None => break,
                }
            }
        }

        // Phase 2: source exhausted. Emit Done. From here on, the velo stream
        // stays open until peer's Done is observed (or the session is killed).
        if let Err(e) = outgoing.send(BidiFrame::Done).await {
            warn!(id = %self.controller.id(), error = %e, "outgoing pump: send Done failed");
            return SessionOutcome::Cancelled;
        }
        trace!(id = %self.controller.id(), "outgoing pump: Done sent, waiting for peer Done");

        // Phase 3: wait for peer's Done before finalizing.
        tokio::select! {
            biased;
            _ = self.controller.killed() => {
                debug!(id = %self.controller.id(), "outgoing pump: controller killed while awaiting peer Done");
                drop(outgoing);
                return SessionOutcome::Cancelled;
            }
            _ = cancel.cancelled() => {
                debug!(id = %self.controller.id(), "outgoing pump: peer cancelled while awaiting Done");
                drop(outgoing);
                return SessionOutcome::Cancelled;
            }
            _ = self.peer_done_observed.notified() => {}
        }

        if let Err(e) = outgoing.finalize() {
            warn!(id = %self.controller.id(), error = %e, "outgoing pump: finalize failed");
            return SessionOutcome::Cancelled;
        }

        debug!(id = %self.controller.id(), "outgoing pump: finalized cleanly");
        SessionOutcome::Finalized
    }
}

/// RAII guard: calls `controller.kill()` on drop. Embed in any struct that
/// owns the user-visible response stream so dropping the stream tears down
/// the rest of the session (the upstream pump aborts, the anchor and sender
/// drop, and velo cancels cross-network).
pub struct SessionGuard {
    controller: Arc<Controller>,
}

impl SessionGuard {
    pub fn new(controller: Arc<Controller>) -> Self {
        Self { controller }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        // Idempotent: Controller::kill is fine to call multiple times.
        <Controller as AsyncEngineContext>::kill(&self.controller);
    }
}

/// Convert a velo `StreamAnchor<BidiFrame<T>>` into a flat `DataStream<T>`.
///
/// Spawns an anchor-drainer task that owns the `StreamAnchor` for its full
/// lifetime (until `Finalized`, an unrecoverable error, or `controller.kill`).
/// User-visible items flow through a tokio mpsc; the user-visible stream
/// ends on the peer's `BidiFrame::Done`, but the drainer keeps polling the
/// anchor so it sees `Finalized` and `terminated=true` before drop — that
/// stops `StreamAnchor::Drop` from emitting a spurious `_stream_cancel` AM
/// that would race with the peer's in-flight `Data` items in the OTHER
/// direction.
///
/// - `BidiFrame::Data(t)` is forwarded to the user-visible mpsc.
/// - `BidiFrame::Done` fires `peer_done` (so our local outgoing pump can
///   finalize) and closes the user-visible mpsc (drainer keeps running).
/// - `StreamFrame::Finalized` ends the drainer.
/// - `controller.killed` cancels the drainer; anchor drop then propagates a
///   normal cancel to the peer (which is what we want for user-cancel).
pub fn decode_bidi_anchor<T>(
    anchor: StreamAnchor<BidiFrame<T>>,
    peer_done: Arc<Notify>,
    controller: Arc<Controller>,
) -> DataStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<T>(16);
    tokio::spawn(drive_bidi_anchor(anchor, peer_done, controller, tx));
    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

async fn drive_bidi_anchor<T>(
    mut anchor: StreamAnchor<BidiFrame<T>>,
    peer_done: Arc<Notify>,
    controller: Arc<Controller>,
    tx: tokio::sync::mpsc::Sender<T>,
) where
    T: DeserializeOwned + Send + 'static,
{
    let mut tx_opt: Option<tokio::sync::mpsc::Sender<T>> = Some(tx);
    let mut peer_done_fired = false;

    loop {
        let frame = tokio::select! {
            biased;
            _ = controller.killed() => {
                debug!(id = %controller.id(), "bidi anchor: controller killed; aborting drainer");
                return;
            }
            f = anchor.next() => f,
        };

        match frame {
            None => {
                // Underlying flume channel closed without a sentinel — the
                // sender's drop should have emitted Dropped first; if not,
                // treat as terminated.
                break;
            }
            Some(Ok(StreamFrame::Item(BidiFrame::Data(t)))) => {
                if let Some(tx) = tx_opt.as_ref()
                    && tx.send(t).await.is_err()
                {
                    // Receiver dropped (user dropped response). Stop
                    // forwarding but keep draining anchor for Finalized
                    // so the anchor terminates cleanly.
                    tx_opt = None;
                }
                // After Done we drop the tx; drop additional Data on the floor.
            }
            Some(Ok(StreamFrame::Item(BidiFrame::Done))) => {
                if !peer_done_fired {
                    peer_done.notify_one();
                    peer_done_fired = true;
                }
                // End the user-visible stream by dropping the sender. The
                // anchor stays alive for further frames (Finalized).
                tx_opt = None;
            }
            Some(Ok(StreamFrame::Finalized)) => {
                if !peer_done_fired {
                    peer_done.notify_one();
                }
                // Anchor's terminated flag is now true; drop is a no-op (no
                // cancel cascade). Drainer ends.
                return;
            }
            Some(Ok(StreamFrame::Detached)) => continue,
            // Heartbeat is filtered inside StreamAnchor before reaching us;
            // Item-shaped sentinels for the other variants are unreachable
            // (velo surfaces them via Err) but match defensively.
            Some(Ok(StreamFrame::Heartbeat))
            | Some(Ok(StreamFrame::SenderError(_)))
            | Some(Ok(StreamFrame::Dropped))
            | Some(Ok(StreamFrame::TransportError(_))) => continue,
            Some(Err(StreamError::SenderError(msg))) => {
                warn!(error = %msg, "bidi anchor: peer soft error, continuing");
                continue;
            }
            Some(Err(e)) => {
                warn!(error = %e, "bidi anchor: stream ended with error");
                if !peer_done_fired {
                    peer_done.notify_one();
                }
                return;
            }
        }
    }
}
