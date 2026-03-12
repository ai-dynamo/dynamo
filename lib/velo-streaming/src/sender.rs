// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`StreamSender<T>`]: typed sender for pushing frames with heartbeat and drop safety.
//!
//! `StreamSender` is the primary write-side abstraction for the streaming protocol.
//! It serializes typed items into [`StreamFrame`] via `rmp_serde`, pushes them through
//! a [`flume::Sender<Vec<u8>>`], manages a background heartbeat task, and guarantees
//! a [`StreamFrame::Dropped`] sentinel on abnormal exit via `impl Drop`.

// Placeholder — tests are written first (TDD RED phase).
// Implementation follows in GREEN phase.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::frame::{SendError, StreamFrame};
    use crate::handle::StreamAnchorHandle;

    use super::StreamSender;

    /// Create a test sender with a bounded(256) channel and a dummy handle.
    fn make_sender() -> (StreamSender<u32>, flume::Receiver<Vec<u8>>) {
        let (tx, rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let sender = StreamSender::new(tx, handle);
        (sender, rx)
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

        let (sender, rx) = make_sender();

        // Advance time past one heartbeat interval (5 seconds)
        tokio::time::advance(Duration::from_secs(6)).await;
        // Yield to let the heartbeat task run
        tokio::task::yield_now().await;

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
        let (sender, rx) = make_sender();

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
        let (sender, rx) = make_sender();

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
        let (sender, rx) = make_sender();

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
        let (sender, rx) = make_sender();
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
        let (sender, rx) = make_sender();

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
        let sender = StreamSender::new(tx, handle);

        // Fill the channel
        let dummy_bytes = rmp_serde::to_vec(&StreamFrame::<()>::Heartbeat).unwrap();
        rx.recv_async().await.ok(); // drain the initial if any

        // Put an item in the channel to fill it
        sender.send(99u32).await.expect("send should succeed");
        // Channel is now full (capacity=1)

        // Advance past heartbeat interval — heartbeat should try_send and not block
        tokio::time::advance(Duration::from_secs(6)).await;
        tokio::task::yield_now().await;

        // The test passes if we get here without hanging — heartbeat didn't block
        // Verify channel still has the original item
        let bytes = rx.try_recv().expect("should have the sent item");
        let frame: StreamFrame<u32> = decode(&bytes);
        assert!(matches!(frame, StreamFrame::Item(99)));

        drop(sender);
        let _ = dummy_bytes;
    }

    // -----------------------------------------------------------------------
    // Test 8: send() on a closed channel returns SendError::ChannelClosed
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_send_on_closed_channel() {
        let (tx, rx) = flume::bounded::<Vec<u8>>(256);
        let handle = StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 1);
        let sender = StreamSender::new(tx, handle);

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

        let (sender, rx) = make_sender();

        // Finalize cancels the heartbeat
        sender.finalize().expect("finalize should succeed");

        // Drain any frames already sent
        while rx.try_recv().is_ok() {}

        // Advance time — no more heartbeats should arrive
        tokio::time::advance(Duration::from_secs(10)).await;
        tokio::task::yield_now().await;

        // Channel should be empty (no heartbeats after cancel)
        assert!(
            rx.try_recv().is_err(),
            "should NOT receive any frames after heartbeat cancel"
        );
    }
}
