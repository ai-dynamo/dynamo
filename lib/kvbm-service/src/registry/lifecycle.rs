// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-stream lifecycle objects held by the [`Registry`](super::Registry).
//!
//! Each active Register call has an associated [`StreamLifecycle`] that owns
//! a clone of the gRPC response sender. The registry uses this to broadcast
//! shutdown events directly onto each client's stream without coupling to
//! the gRPC handler internals.
//!
//! The trait is `pub` because it is the abstraction for "server-side
//! per-client state that should receive a shutdown signal". For the gRPC
//! path the only impl is [`GrpcStreamLifecycle`]. [`NoopLifecycle`] is a
//! useful no-op for tests and for any direct-call paths that don't need
//! per-client shutdown signalling.

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio_util::sync::CancellationToken;
use tonic::Status;
use tracing::{debug, warn};

use crate::proto;
use crate::registry::client::RegistrationId;

/// Per-stream control surface used by the shutdown coordinator.
#[async_trait]
pub trait StreamLifecycle: Send + Sync {
    /// Enqueue a [`proto::ServerShutdownInitiated`] event on the client's
    /// stream. Best-effort: if the receiver is already gone, this is a
    /// no-op. Returns immediately after the send completes (or fails).
    async fn send_shutdown_initiated(&self, grace_period: Option<Duration>);

    /// Enqueue a [`proto::ServerShutdownTimedOut`] event and then drop the
    /// underlying sender, causing the client's stream to close on the next
    /// poll. Idempotent: subsequent calls are no-ops.
    async fn send_shutdown_timed_out(&self, grace_period: Duration);
}

/// Production impl over a tonic mpsc response channel.
///
/// Holds one sender clone and a [`CancellationToken`] that the gRPC handler
/// also shares with the watcher and heartbeat tasks. When
/// [`Self::send_shutdown_timed_out`] runs it drops its sender **and** fires
/// the token so the other tasks drop their senders too — only then will the
/// receiver yield `None` and the client stream close.
pub(crate) struct GrpcStreamLifecycle {
    tx: Mutex<Option<mpsc::Sender<Result<proto::Event, Status>>>>,
    registration_id: RegistrationId,
    stream_cancel: CancellationToken,
}

impl GrpcStreamLifecycle {
    pub(crate) fn new(
        tx: mpsc::Sender<Result<proto::Event, Status>>,
        registration_id: RegistrationId,
        stream_cancel: CancellationToken,
    ) -> Self {
        Self {
            tx: Mutex::new(Some(tx)),
            registration_id,
            stream_cancel,
        }
    }
}

#[async_trait]
impl StreamLifecycle for GrpcStreamLifecycle {
    async fn send_shutdown_initiated(&self, grace_period: Option<Duration>) {
        let grace_period_ms = grace_period.map(duration_to_ms).unwrap_or(0);
        let event = proto::Event {
            kind: Some(proto::event::Kind::ServerShutdownInitiated(
                proto::ServerShutdownInitiated { grace_period_ms },
            )),
        };
        let guard = self.tx.lock().await;
        // Non-blocking send: a client that has stopped reading must not be
        // able to stall the shutdown broadcast by leaving the bounded
        // channel full. We still signal them by other means (the channel
        // closing on the eventual force-drop), so dropping this event is
        // acceptable.
        if let Some(tx) = guard.as_ref() {
            match tx.try_send(Ok(event)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    warn!(
                        registration_id = %self.registration_id,
                        "shutdown_initiated dropped — client stream channel is full",
                    );
                }
                Err(TrySendError::Closed(_)) => {
                    debug!(
                        registration_id = %self.registration_id,
                        "shutdown_initiated skipped — client gone",
                    );
                }
            }
        }
    }

    async fn send_shutdown_timed_out(&self, grace_period: Duration) {
        let event = proto::Event {
            kind: Some(proto::event::Kind::ServerShutdownTimedOut(
                proto::ServerShutdownTimedOut {
                    grace_period_ms: duration_to_ms(grace_period),
                },
            )),
        };
        let mut guard = self.tx.lock().await;
        let Some(tx) = guard.take() else {
            // Already forced — nothing to do.
            return;
        };
        // Non-blocking send: cancelling the stream must not be gated by
        // channel capacity. If the event can't be delivered, we still drop
        // the sender and fire the cancel token below so the stream closes.
        match tx.try_send(Ok(event)) {
            Ok(()) => warn!(
                registration_id = %self.registration_id,
                grace_ms = duration_to_ms(grace_period),
                "grace period elapsed; forcing stream close",
            ),
            Err(TrySendError::Full(_)) => warn!(
                registration_id = %self.registration_id,
                "grace period elapsed; channel full, force-closing without TimedOut event",
            ),
            Err(TrySendError::Closed(_)) => debug!(
                registration_id = %self.registration_id,
                "grace period elapsed; client already disconnected",
            ),
        }
        // `tx` drops here. The receiver still won't close until the watcher
        // and heartbeat tasks also drop their sender clones — signal them
        // via the shared cancellation token.
        drop(tx);
        self.stream_cancel.cancel();
    }
}

/// No-op [`StreamLifecycle`] for tests and for paths that don't need
/// per-client shutdown signalling.
pub struct NoopLifecycle;

#[async_trait]
impl StreamLifecycle for NoopLifecycle {
    async fn send_shutdown_initiated(&self, _grace_period: Option<Duration>) {}
    async fn send_shutdown_timed_out(&self, _grace_period: Duration) {}
}

fn duration_to_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;

    fn make_lifecycle() -> (
        GrpcStreamLifecycle,
        ReceiverStream<Result<proto::Event, Status>>,
        CancellationToken,
    ) {
        let (tx, rx) = mpsc::channel(8);
        let token = CancellationToken::new();
        let lc = GrpcStreamLifecycle::new(tx, RegistrationId::new(), token.clone());
        (lc, ReceiverStream::new(rx), token)
    }

    #[tokio::test]
    async fn shutdown_initiated_carries_grace() {
        let (lc, mut stream, _token) = make_lifecycle();
        lc.send_shutdown_initiated(Some(Duration::from_secs(90)))
            .await;
        let ev = stream.next().await.unwrap().unwrap();
        let proto::event::Kind::ServerShutdownInitiated(body) = ev.kind.unwrap() else {
            panic!("expected ServerShutdownInitiated");
        };
        assert_eq!(body.grace_period_ms, 90_000);
    }

    #[tokio::test]
    async fn shutdown_initiated_with_none_grace_sends_zero() {
        let (lc, mut stream, _token) = make_lifecycle();
        lc.send_shutdown_initiated(None).await;
        let ev = stream.next().await.unwrap().unwrap();
        let proto::event::Kind::ServerShutdownInitiated(body) = ev.kind.unwrap() else {
            panic!("expected ServerShutdownInitiated");
        };
        assert_eq!(body.grace_period_ms, 0);
    }

    #[tokio::test]
    async fn shutdown_timed_out_closes_stream_and_fires_cancel() {
        let (lc, mut stream, token) = make_lifecycle();
        assert!(!token.is_cancelled());
        lc.send_shutdown_timed_out(Duration::from_secs(60)).await;
        let ev = stream.next().await.unwrap().unwrap();
        let proto::event::Kind::ServerShutdownTimedOut(body) = ev.kind.unwrap() else {
            panic!("expected ServerShutdownTimedOut");
        };
        assert_eq!(body.grace_period_ms, 60_000);
        // The sender was taken+dropped; the receiver yields None on next poll.
        assert!(
            stream.next().await.is_none(),
            "stream must close after timed-out event"
        );
        // And the cancellation token was fired so any sibling tasks
        // (watcher, heartbeat) drop their sender clones too.
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn shutdown_timed_out_is_idempotent() {
        let (lc, mut stream, _token) = make_lifecycle();
        lc.send_shutdown_timed_out(Duration::from_secs(60)).await;
        // Second call is a no-op (sender already taken).
        lc.send_shutdown_timed_out(Duration::from_secs(60)).await;
        // Drain: one event, then stream closes.
        let _ = stream.next().await.unwrap().unwrap();
        assert!(stream.next().await.is_none());
    }
}
