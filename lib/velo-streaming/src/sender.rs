// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StreamSender<T>`]: typed sender for pushing frames with heartbeat and drop safety.
//!
//! `StreamSender` is the primary write-side abstraction for the streaming protocol.
//! It serializes typed items into [`StreamFrame`] via `rmp_serde`, pushes them through
//! a [`flume::Sender<Vec<u8>>`], manages a background heartbeat task, and guarantees
//! a [`StreamFrame::Dropped`] sentinel on abnormal exit via `impl Drop`.
//!
//! # Lifecycle
//!
//! A `StreamSender<T>` is created by [`AnchorManager::attach_stream_anchor`] and
//! must be terminated via one of three paths:
//!
//! 1. **[`finalize(self)`](StreamSender::finalize)** — clean close, sends `Finalized` sentinel.
//! 2. **[`detach(self)`](StreamSender::detach)** — clean detach, sends `Detached` sentinel, returns handle for re-attach.
//! 3. **Drop** — abnormal exit, sends `Dropped` sentinel synchronously.
//!
//! The heartbeat background task is cancelled in all three paths before the
//! terminal sentinel is sent.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::anchor::AnchorEntry;
use crate::frame::{SendError, StreamFrame};
use crate::handle::StreamAnchorHandle;

/// Heartbeat interval: emits a [`StreamFrame::Heartbeat`] every 5 seconds.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

/// Typed sender for pushing frames through the streaming channel.
///
/// Holds a [`flume::Sender<Vec<u8>>`] for serialized frame bytes, a
/// [`StreamAnchorHandle`] identifying the anchor, a [`CancellationToken`]
/// to stop the background heartbeat task, and a reference to the anchor
/// registry so that [`detach`](StreamSender::detach) can atomically clear
/// the attachment flag before returning the handle.
///
/// `T` is the user-defined item payload type. The `Serialize` bound is required
/// for [`send`](StreamSender::send) to serialize `StreamFrame::Item(T)`.
/// Sentinel methods (`finalize`, `detach`, Drop) use `StreamFrame::<()>` to
/// avoid the `Serialize` bound (sentinel variants carry no `T` data).
pub struct StreamSender<T> {
    tx: flume::Sender<Vec<u8>>,
    handle: StreamAnchorHandle,
    heartbeat_cancel: CancellationToken,
    sent_terminal: bool,
    /// Registry reference for clearing the attachment flag on detach.
    registry: Arc<DashMap<u64, AnchorEntry>>,
    // --- Phase 11 additions ---
    /// User-facing cancellation signal: fires when _stream_cancel is received.
    cancel_token: CancellationToken,
    /// Key in the sender-side registry for cleanup and for the _stream_cancel handler.
    sender_stream_id: u64,
    /// Sender-side registry shared with the _stream_cancel handler.
    sender_registry: Arc<crate::control::SenderRegistry>,
    /// Poison channel sender: when rx_closer (the receiver) is dropped by _stream_cancel handler,
    /// this becomes disconnected and send() returns ChannelClosed.
    poison_tx: flume::Sender<()>,
    // --- end Phase 11 ---
    _phantom: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for StreamSender<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSender")
            .field("handle", &self.handle)
            .field("sent_terminal", &self.sent_terminal)
            .finish_non_exhaustive()
    }
}


impl<T: Serialize> StreamSender<T> {
    /// Create a new `StreamSender` and spawn the background heartbeat task.
    ///
    /// The heartbeat task emits [`StreamFrame::Heartbeat`] every 5 seconds via
    /// non-blocking `try_send`. It is cancelled when the sender is finalized,
    /// detached, or dropped.
    ///
    /// `registry` is a shared reference to the anchor registry so that
    /// [`detach`](StreamSender::detach) can atomically clear the attachment flag.
    pub(crate) fn new(
        tx: flume::Sender<Vec<u8>>,
        handle: StreamAnchorHandle,
        registry: Arc<DashMap<u64, AnchorEntry>>,
        cancel_token: CancellationToken,
        sender_stream_id: u64,
        sender_registry: Arc<crate::control::SenderRegistry>,
        poison_tx: flume::Sender<()>,
    ) -> Self {
        let heartbeat_cancel = CancellationToken::new();

        // Spawn heartbeat background task
        let cancel = heartbeat_cancel.clone();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
            // Skip the first immediate tick
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = interval.tick() => {
                        // Serialize Heartbeat using StreamFrame::<()> — the phantom
                        // type doesn't matter for sentinel-only variants since the
                        // msgpack encoding is identical regardless of T.
                        let bytes = rmp_serde::to_vec(&StreamFrame::<()>::Heartbeat)
                            .expect("Heartbeat serializes infallibly");
                        // Non-blocking try_send: if the channel is full we silently
                        // drop the heartbeat rather than stalling the sender.
                        let _ = tx_clone.try_send(bytes);
                    }
                }
            }
        });

        Self {
            tx,
            handle,
            heartbeat_cancel,
            sent_terminal: false,
            registry,
            cancel_token,
            sender_stream_id,
            sender_registry,
            poison_tx,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a cloneable `CancellationToken` that fires when the consumer
    /// cancels the stream (e.g. drops `StreamAnchor` or calls `StreamController::cancel()`).
    ///
    /// Use in a `tokio::select!` to stop production proactively:
    /// ```no_run
    /// # async fn example() {
    /// # let mut sender: velo_streaming::StreamSender<u32> = todo!();
    /// # async fn produce() -> u32 { 0 }
    /// loop {
    ///     tokio::select! {
    ///         _ = sender.cancellation_token().cancelled() => break,
    ///         val = produce() => { let _ = sender.send(val).await; }
    ///     }
    /// }
    /// # }
    /// ```
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Send a typed item through the channel.
    ///
    /// Serializes `StreamFrame::Item(item)` via `rmp_serde` and pushes the
    /// resulting bytes through the flume channel asynchronously.
    ///
    /// # Errors
    ///
    /// - [`SendError::SerializationError`] if `rmp_serde::to_vec` fails.
    /// - [`SendError::ChannelClosed`] if the receiver has been dropped.
    pub async fn send(&self, item: T) -> Result<(), SendError> {
        // Check if the poison channel has been disconnected (rx_closer dropped by _stream_cancel handler).
        if self.poison_tx.is_disconnected() {
            return Err(SendError::ChannelClosed);
        }
        let bytes = rmp_serde::to_vec(&StreamFrame::Item(item))
            .map_err(|e| SendError::SerializationError(e.to_string()))?;
        self.tx
            .send_async(bytes)
            .await
            .map_err(|_| SendError::ChannelClosed)
    }

    /// Send a soft error through the channel.
    ///
    /// Serializes `StreamFrame::SenderError(msg)` and pushes via `send_async`.
    /// Uses `StreamFrame::<()>::SenderError` to avoid requiring `T: Serialize`
    /// just for error sending -- the SenderError variant carries only a String,
    /// so the msgpack encoding is identical regardless of the phantom type `T`.
    ///
    /// # Errors
    ///
    /// - [`SendError::ChannelClosed`] if the receiver has been dropped.
    pub async fn send_err(&self, msg: impl ToString) -> Result<(), SendError> {
        // Safe to use StreamFrame::<()> here: the SenderError variant carries
        // only a String and its msgpack encoding is identical for any T.
        let bytes = rmp_serde::to_vec(&StreamFrame::<()>::SenderError(msg.to_string()))
            .expect("SenderError serializes infallibly");
        self.tx
            .send_async(bytes)
            .await
            .map_err(|_| SendError::ChannelClosed)
    }

    /// Permanently close the stream by sending a `Finalized` sentinel.
    ///
    /// Cancels the heartbeat task, sends `StreamFrame::Finalized` synchronously,
    /// and consumes `self`. The subsequent `Drop` will see `sent_terminal = true`
    /// and skip the `Dropped` sentinel.
    ///
    /// # Errors
    ///
    /// - [`SendError::ChannelClosed`] if the receiver has already been dropped.
    pub fn finalize(mut self) -> Result<(), SendError> {
        self.heartbeat_cancel.cancel();
        let bytes = rmp_serde::to_vec(&StreamFrame::<()>::Finalized)
            .expect("Finalized serializes infallibly");
        self.sent_terminal = true;
        // Clean up sender registry entry before returning
        self.sender_registry.senders.remove(&self.sender_stream_id);
        self.tx.send(bytes).map_err(|_| SendError::ChannelClosed)
    }

    /// Detach the sender from the anchor by sending a `Detached` sentinel.
    ///
    /// Cancels the heartbeat task, sends `StreamFrame::Detached` synchronously,
    /// clears the attachment flag in the registry (so a new sender can attach
    /// immediately), and returns the [`StreamAnchorHandle`] for re-attachment.
    /// The subsequent `Drop` will see `sent_terminal = true` and skip `Dropped`.
    ///
    /// # Errors
    ///
    /// - [`SendError::ChannelClosed`] if the receiver has already been dropped.
    pub fn detach(mut self) -> Result<StreamAnchorHandle, SendError> {
        self.heartbeat_cancel.cancel();
        let bytes = rmp_serde::to_vec(&StreamFrame::<()>::Detached)
            .expect("Detached serializes infallibly");
        self.sent_terminal = true;
        self.tx.send(bytes).map_err(|_| SendError::ChannelClosed)?;
        // Atomically clear the attachment flag so a new sender can attach
        // without waiting for AnchorStream to consume the Detached sentinel.
        let (_, local_id) = self.handle.unpack();
        if let Some(mut entry) = self.registry.get_mut(&local_id) {
            entry.attachment = false;
        }
        // Clean up sender registry entry
        self.sender_registry.senders.remove(&self.sender_stream_id);
        Ok(self.handle)
    }
}

/// Drop safety: sends `StreamFrame::Dropped` synchronously if no terminal was sent.
///
/// This `impl` block has no `T: Serialize` bound because sentinel serialization
/// uses `StreamFrame::<()>` — Rust forbids trait bounds on `Drop` impls.
impl<T> Drop for StreamSender<T> {
    fn drop(&mut self) {
        // Always clean up sender registry entry to prevent memory leak.
        // This is idempotent: if finalize/detach already removed it, this is a no-op.
        self.sender_registry.senders.remove(&self.sender_stream_id);
        if !self.sent_terminal {
            self.heartbeat_cancel.cancel();
            let bytes = rmp_serde::to_vec(&StreamFrame::<()>::Dropped)
                .expect("Dropped serializes infallibly");
            // Synchronous send — no spawn, no block_on. Ignore errors (channel
            // may already be closed if the receiver was dropped first).
            let _ = self.tx.send(bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use dashmap::DashMap;

    use crate::anchor::AnchorEntry;
    use crate::frame::{SendError, StreamFrame};
    use crate::handle::StreamAnchorHandle;

    use super::StreamSender;

    /// Create an empty registry for use in unit tests (no real anchors needed).
    fn empty_registry() -> Arc<DashMap<u64, AnchorEntry>> {
        Arc::new(DashMap::new())
    }

    /// Create a test sender with a bounded(256) channel and a dummy handle.
    ///
    /// Returns `(sender, frame_rx, _poison_rx)`.
    /// The `_poison_rx` must be kept alive for the lifetime of the sender —
    /// dropping it simulates `_stream_cancel` and causes `send()` to return `ChannelClosed`.
    fn make_sender() -> (StreamSender<u32>, flume::Receiver<Vec<u8>>, flume::Receiver<()>) {
        let (tx, rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let sender_registry = std::sync::Arc::new(crate::control::SenderRegistry::default());
        let (poison_tx, poison_rx) = flume::bounded::<()>(1);
        let sender = StreamSender::new(tx, handle, empty_registry(), cancel_token, 1, sender_registry, poison_tx);
        (sender, rx, poison_rx)
    }

    /// Create a sender entry and insert it into a registry. Returns (registry, sender_stream_id).
    fn make_sender_with_registry(
        tx: flume::Sender<Vec<u8>>,
        handle: StreamAnchorHandle,
        sender_stream_id: u64,
    ) -> (StreamSender<u32>, std::sync::Arc<crate::control::SenderRegistry>) {
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let sender_registry = std::sync::Arc::new(crate::control::SenderRegistry::default());
        let (poison_tx, poison_rx) = flume::bounded::<()>(1);

        // Insert the SenderEntry into the registry (simulating what attach_stream_anchor does)
        let entry = crate::control::SenderEntry {
            cancel_token: cancel_token.clone(),
            rx_closer: std::sync::Mutex::new(Some(poison_rx)),
        };
        sender_registry.senders.insert(sender_stream_id, entry);

        let sender = StreamSender::new(
            tx,
            handle,
            empty_registry(),
            cancel_token,
            sender_stream_id,
            sender_registry.clone(),
            poison_tx,
        );
        (sender, sender_registry)
    }

    /// Helper: deserialize raw bytes into StreamFrame<T>.
    fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> StreamFrame<T> {
        rmp_serde::from_slice(bytes).expect("deserialize StreamFrame")
    }

    // -----------------------------------------------------------------------
    // Test 1: StreamSender::new() spawns a heartbeat task
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_heartbeat_emits() {
        tokio::time::pause();

        let (sender, rx, _poison_rx) = make_sender();

        // Advance time past one heartbeat interval (5 seconds).
        // Use sleep rather than advance+yield so the interval task gets polled.
        tokio::time::sleep(Duration::from_secs(6)).await;

        // Should have received at least one heartbeat
        let bytes = rx.try_recv().expect("should receive heartbeat frame");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(
            matches!(frame, StreamFrame::Heartbeat),
            "expected Heartbeat, got {:?}",
            frame
        );

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 2: send(item) serializes StreamFrame::Item(item) and sends
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_send_item() {
        let (sender, rx, _poison_rx) = make_sender();

        sender.send(42u32).await.expect("send should succeed");

        let bytes = rx.recv_async().await.expect("should receive item");
        let frame: StreamFrame<u32> = decode(&bytes);
        match frame {
            StreamFrame::Item(val) => assert_eq!(val, 42),
            other => panic!("expected Item(42), got {:?}", other),
        }

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 3: send_err("msg") serializes StreamFrame::SenderError
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_send_err() {
        let (sender, rx, _poison_rx) = make_sender();

        sender
            .send_err("something went wrong")
            .await
            .expect("send_err should succeed");

        let bytes = rx.recv_async().await.expect("should receive error frame");
        let frame: StreamFrame<u32> = decode(&bytes);
        match frame {
            StreamFrame::SenderError(msg) => assert_eq!(msg, "something went wrong"),
            other => panic!("expected SenderError, got {:?}", other),
        }

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 4: finalize(self) sends Finalized, cancels heartbeat, no Dropped
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_finalize() {
        let (sender, rx, _poison_rx) = make_sender();

        sender.finalize().expect("finalize should succeed");

        let bytes = rx.recv_async().await.expect("should receive Finalized");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(
            matches!(frame, StreamFrame::Finalized),
            "expected Finalized, got {:?}",
            frame
        );

        // No Dropped should follow — drain any heartbeats and check
        while let Ok(bytes) = rx.try_recv() {
            let frame: StreamFrame<u32> = decode(&bytes);
            assert!(
                !matches!(frame, StreamFrame::Dropped),
                "should NOT receive Dropped after finalize"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: detach(self) sends Detached, returns handle, no Dropped
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_detach() {
        let (sender, rx, _poison_rx) = make_sender();
        let expected_handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);

        let returned_handle = sender.detach().expect("detach should succeed");
        assert_eq!(returned_handle, expected_handle);

        let bytes = rx.recv_async().await.expect("should receive Detached");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(
            matches!(frame, StreamFrame::Detached),
            "expected Detached, got {:?}",
            frame
        );

        // No Dropped should follow
        while let Ok(bytes) = rx.try_recv() {
            let frame: StreamFrame<u32> = decode(&bytes);
            assert!(
                !matches!(frame, StreamFrame::Dropped),
                "should NOT receive Dropped after detach"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: dropping StreamSender without terminal sends Dropped
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_drop_sends_dropped() {
        let (sender, rx, _poison_rx) = make_sender();

        // Drop without finalize or detach
        drop(sender);

        // Should receive Dropped
        let bytes = rx.recv_async().await.expect("should receive Dropped");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(
            matches!(frame, StreamFrame::Dropped),
            "expected Dropped, got {:?}",
            frame
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: heartbeat uses try_send (non-blocking) -- doesn't block on full channel
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_heartbeat_non_blocking_on_full_channel() {
        tokio::time::pause();

        // Create a channel with capacity 1
        let (tx, rx) = flume::bounded::<Vec<u8>>(1);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let sender_registry = std::sync::Arc::new(crate::control::SenderRegistry::default());
        let (poison_tx, _poison_rx) = flume::bounded::<()>(1);
        let sender = StreamSender::new(tx, handle, empty_registry(), cancel_token, 1, sender_registry, poison_tx);

        // Put an item in the channel to fill it
        sender.send(99u32).await.expect("send should succeed");
        // Channel is now full (capacity=1)

        // Advance past heartbeat interval — heartbeat should try_send and not block.
        // Use sleep so the interval task gets polled under paused time.
        tokio::time::sleep(Duration::from_secs(6)).await;

        // The test passes if we get here without hanging — heartbeat didn't block
        // Verify channel still has the original item
        let bytes = rx.try_recv().expect("should have the sent item");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(matches!(frame, StreamFrame::Item(99)));

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 8: send() on a closed channel returns SendError::ChannelClosed
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_send_on_closed_channel() {
        let (tx, rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let sender_registry = std::sync::Arc::new(crate::control::SenderRegistry::default());
        let (poison_tx, _poison_rx) = flume::bounded::<()>(1);
        let sender = StreamSender::new(tx, handle, empty_registry(), cancel_token, 1, sender_registry, poison_tx);

        // Drop the receiver to close the channel
        drop(rx);

        let result = sender.send(42u32).await;
        assert!(
            matches!(result, Err(SendError::ChannelClosed)),
            "expected ChannelClosed, got {:?}",
            result
        );

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 9: heartbeat task stops after cancel (CancellationToken)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_heartbeat_stops_after_cancel() {
        tokio::time::pause();

        let (sender, rx, _poison_rx) = make_sender();

        // Finalize cancels the heartbeat
        sender.finalize().expect("finalize should succeed");

        // Drain any frames already sent
        while rx.try_recv().is_ok() {}

        // Advance time — no more heartbeats should arrive.
        // Use sleep so any pending tasks get polled under paused time.
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Channel should be empty (no heartbeats after cancel)
        assert!(
            rx.try_recv().is_err(),
            "should NOT receive any frames after heartbeat cancel"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: cancellation_token() returns a cloneable token
    // -----------------------------------------------------------------------

    #[test]
    fn test_cancellation_token() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (sender, _rx, _poison_rx) = make_sender();
            let token = sender.cancellation_token();
            // Token should not be cancelled yet
            assert!(!token.is_cancelled(), "token should not start cancelled");
            // Cancelling the token externally should be reflected
            token.cancel();
            assert!(token.is_cancelled(), "token should be cancelled after cancel()");
            // A clone should also be cancelled
            let clone = sender.cancellation_token();
            assert!(clone.is_cancelled(), "cloned token should reflect cancellation");
            drop(sender);
        });
    }

    // -----------------------------------------------------------------------
    // Test 11: send() returns ChannelClosed after poison_tx disconnected
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_send_after_cancel() {
        let (frame_tx, _frame_rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 2);

        // Create a poison channel and keep track of the receiver
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let sender_registry = std::sync::Arc::new(crate::control::SenderRegistry::default());
        let (poison_tx, poison_rx) = flume::bounded::<()>(1);

        let sender = StreamSender::<u32>::new(
            frame_tx,
            handle,
            empty_registry(),
            cancel_token,
            2,
            sender_registry,
            poison_tx,
        );

        // Drop the receiver (simulating rx_closer drop in _stream_cancel handler)
        drop(poison_rx);

        // send() should now return ChannelClosed
        let result = sender.send(42u32).await;
        assert!(
            matches!(result, Err(SendError::ChannelClosed)),
            "expected ChannelClosed after poison_rx drop, got {:?}",
            result
        );

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // Test 12: SenderRegistry entry removed after finalize()
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_registry_cleanup_on_finalize() {
        let (tx, _rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let (sender, registry) = make_sender_with_registry(tx, handle, 1);

        // Entry should be present before finalize
        assert!(registry.senders.contains_key(&1), "entry should be present before finalize");

        sender.finalize().expect("finalize should succeed");

        // Entry should be removed after finalize
        assert!(
            !registry.senders.contains_key(&1),
            "entry should be removed after finalize"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13: SenderRegistry entry removed after detach()
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_registry_cleanup_on_detach() {
        let (tx, _rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let (sender, registry) = make_sender_with_registry(tx, handle, 1);

        // Entry should be present before detach
        assert!(registry.senders.contains_key(&1), "entry should be present before detach");

        sender.detach().expect("detach should succeed");

        // Entry should be removed after detach
        assert!(
            !registry.senders.contains_key(&1),
            "entry should be removed after detach"
        );
    }

    // -----------------------------------------------------------------------
    // Test 14: SenderRegistry entry removed after drop
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_cleanup_on_drop() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let (tx, _rx) = flume::bounded::<Vec<u8>>(256);
            let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
            let (sender, registry) = make_sender_with_registry(tx, handle, 1);

            // Entry should be present before drop
            assert!(registry.senders.contains_key(&1), "entry should be present before drop");

            // Drop without finalize or detach
            drop(sender);

            // Entry should be removed after drop
            assert!(
                !registry.senders.contains_key(&1),
                "entry should be removed after drop"
            );
        });
    }
}
