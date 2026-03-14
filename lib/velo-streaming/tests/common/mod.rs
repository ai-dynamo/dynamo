// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test infrastructure for velo-streaming integration tests.
//!
//! Provides:
//! - [`MockFrameTransport`]: in-memory transport implementing [`FrameTransport`] for unit/integration tests
//! - [`run_transport_tests!`]: macro that expands to a test module with TEST-11 and TEST-13 scenarios
//! - [`make_mock_manager`]: helper to create an [`AnchorManager`] backed by [`MockFrameTransport`]

use std::sync::Arc;

use futures::future::BoxFuture;
use velo_common::WorkerId;
use velo_streaming::{AnchorManager, FrameTransport};

/// In-memory [`FrameTransport`] implementation for local integration tests.
///
/// `bind()` returns a fresh `flume::bounded(256)` receiver; the sender half is immediately
/// dropped because `StreamSender` writes frames directly to `StreamAnchor` via the internal
/// `frame_tx` clone — not through the transport channel. The transport channel is only used
/// by the `reader_pump` task, which is not exercised in local mock tests.
///
/// `connect()` returns a disconnected sender — also unused by `StreamSender` in local tests.
pub struct MockFrameTransport;

impl MockFrameTransport {
    pub fn new() -> Self {
        Self
    }
}

impl FrameTransport for MockFrameTransport {
    fn bind(&self, anchor_id: u64) -> BoxFuture<'_, anyhow::Result<(String, flume::Receiver<Vec<u8>>)>> {
        Box::pin(async move {
            let (_tx, rx) = flume::bounded::<Vec<u8>>(256);
            // tx dropped immediately — bind receiver is for reader_pump only.
            // StreamSender writes to frame_tx (the StreamAnchor's internal channel),
            // not to this transport channel.
            Ok((format!("mock://{}", anchor_id), rx))
        })
    }

    fn connect(
        &self,
        _endpoint: &str,
        _anchor_id: u64,
        _worker_id: u64,
    ) -> BoxFuture<'_, anyhow::Result<flume::Sender<Vec<u8>>>> {
        Box::pin(async move {
            // StreamSender writes to frame_tx (not this sender), so returning a
            // disconnected sender is safe for local/mock tests.
            let (tx, _rx) = flume::bounded::<Vec<u8>>(1);
            Ok(tx)
        })
    }
}

/// Create an [`AnchorManager`] backed by [`MockFrameTransport`].
pub async fn make_mock_manager() -> Arc<AnchorManager> {
    Arc::new(AnchorManager::new(
        WorkerId::from_u64(1),
        Arc::new(MockFrameTransport::new()),
    ))
}

/// Expand to a test module containing integration scenarios for the streaming protocol.
///
/// # Parameters
/// - `$mod_name`: identifier for the generated module (e.g. `mock_suite`)
/// - `$make_manager`: expression that evaluates to `Arc<AnchorManager>`
///
/// # Tests generated
/// - `test_01_local_round_trip`: round-trip 10 items + finalize
/// - `test_04_detach_reattach`: detach/reattach with two senders; all 6 items delivered
/// - `test_05_finalize_closes_stream`: finalize drains stream; registry entry removed
/// - `test_06_cancel_prevents_attach`: cancel removes anchor; attach returns AnchorNotFound
/// - `test_08_drop_safety`: sender dropped without detach/finalize sends Dropped sentinel
/// - `test_11_mock_transport_full_cycle`: create anchor, attach, send 5 items, finalize, collect
/// - `test_12_sentinel_ordering`: 1000 items + sender drop; ordering guarantees verified
/// - `test_13_unit_coverage`: handle pack/unpack roundtrip, monotonic IDs, exclusive attach
#[macro_export]
macro_rules! run_transport_tests {
    ($mod_name:ident, $make_manager:expr) => {
        mod $mod_name {
            use super::*;
            use futures::StreamExt;
            use std::sync::Arc;
            use velo_common::WorkerId;
            use velo_streaming::{AnchorManager, AttachError, StreamAnchorHandle, StreamError, StreamFrame};

            async fn manager() -> Arc<AnchorManager> {
                $make_manager
            }

            /// TEST-01: Local round-trip — send 10 items + finalize; stream yields all 10 then None.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_01_local_round_trip() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender = mgr.attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("attach must succeed");
                for i in 0u32..10 {
                    sender.send(i).await.expect("send");
                }
                sender.finalize().expect("finalize");
                let mut items = Vec::new();
                while let Some(frame) = anchor.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => items.push(v),
                        Ok(StreamFrame::Finalized) => break,
                        other => panic!("unexpected frame: {:?}", other),
                    }
                }
                assert_eq!(items, (0u32..10).collect::<Vec<_>>());
                assert!(anchor.next().await.is_none());
            }

            /// TEST-04: Detach/reattach — first sender sends 3, detaches; second sends 3 more,
            /// finalizes; stream yields all 6 items in order with a Detached sentinel between batches.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_04_detach_reattach() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender1 = mgr.attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("first attach");
                for i in 0u32..3 {
                    sender1.send(i).await.expect("send first batch");
                }
                // Detach returns the handle for reattachment
                let returned_handle = sender1.detach().expect("detach must succeed");
                // After detach, stream should yield Detached frame then stay open
                // Reattach with the returned handle
                let sender2 = mgr.attach_stream_anchor::<u32>(returned_handle)
                    .await
                    .expect("second attach");
                for i in 3u32..6 {
                    sender2.send(i).await.expect("send second batch");
                }
                sender2.finalize().expect("finalize");
                // Collect all frames
                let mut items = Vec::new();
                while let Some(frame) = anchor.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => items.push(v),
                        Ok(StreamFrame::Detached) => { /* detach sentinel between batches */ }
                        Ok(StreamFrame::Finalized) => break,
                        other => panic!("unexpected frame: {:?}", other),
                    }
                }
                assert_eq!(items, vec![0u32, 1, 2, 3, 4, 5]);
                assert!(anchor.next().await.is_none());
            }

            /// TEST-05: Finalize closes stream — stream drains items then yields None;
            /// anchor is removed from registry after finalize.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_05_finalize_closes_stream() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender = mgr.attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("attach");
                for i in 0u32..5 {
                    sender.send(i).await.expect("send");
                }
                sender.finalize().expect("finalize");
                let mut items = Vec::new();
                while let Some(frame) = anchor.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => items.push(v),
                        Ok(StreamFrame::Finalized) => break,
                        other => panic!("unexpected: {:?}", other),
                    }
                }
                assert_eq!(items.len(), 5);
                // Stream yields None after Finalized
                assert!(anchor.next().await.is_none());
                // Registry entry removed after finalize; second attach must fail
                let second_attach = mgr.attach_stream_anchor::<u32>(handle).await;
                assert!(
                    matches!(second_attach, Err(velo_streaming::AttachError::AnchorNotFound { .. })),
                    "anchor must be absent from registry after finalize"
                );
            }

            /// TEST-06: Cancel prevents attach — after anchor.cancel(), attach returns AnchorNotFound.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_06_cancel_prevents_attach() {
                let mgr = manager().await;
                let anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                anchor.cancel();
                // Give tokio a chance to process cancel
                tokio::task::yield_now().await;
                let result = mgr.attach_stream_anchor::<u32>(handle).await;
                assert!(
                    matches!(result, Err(velo_streaming::AttachError::AnchorNotFound { .. })),
                    "attach after cancel must return AnchorNotFound"
                );
            }

            /// TEST-08: Drop safety — sender dropped without detach/finalize sends Dropped sentinel;
            /// stream yields Err(SenderDropped) then None.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_08_drop_safety() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender = mgr.attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("attach");
                sender.send(42u32).await.expect("send one item");
                // Drop sender WITHOUT explicit detach or finalize.
                // impl Drop sends StreamFrame::Dropped synchronously before returning from drop().
                drop(sender);
                // Collect frames: expect Item(42) then Err(SenderDropped) then None
                let frame1 = anchor.next().await;
                assert!(
                    matches!(frame1, Some(Ok(StreamFrame::Item(42u32)))),
                    "first frame must be Item(42), got {:?}", frame1
                );
                let frame2 = anchor.next().await;
                assert!(
                    matches!(frame2, Some(Err(velo_streaming::StreamError::SenderDropped))),
                    "second frame must be Err(SenderDropped) from Dropped sentinel, got {:?}", frame2
                );
                assert!(anchor.next().await.is_none(), "stream must be exhausted after Dropped");
            }

            /// TEST-12: Sentinel ordering — 1000 items sent then sender dropped; assert ordering
            /// guarantees: (a) exactly 1000 Item frames, (b) Dropped is the last frame, (c) no
            /// Item frames follow the Dropped sentinel.
            ///
            /// Sender runs in a spawned task so it can block on the bounded channel (256)
            /// while the main task concurrently drains the stream.
            #[tokio::test(flavor = "multi_thread")]
            async fn test_12_sentinel_ordering() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender = mgr.attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("attach");
                // Spawn sender: sends 1000 items then drops (Dropped sent synchronously in drop())
                let send_task = tokio::spawn(async move {
                    for i in 0u32..1000 {
                        sender.send(i).await.expect("send");
                    }
                    // sender dropped here -> Dropped sentinel sent synchronously
                });
                // Collect ALL frames until None (stream drains concurrently with sender)
                let mut items: Vec<u32> = Vec::new();
                let mut saw_dropped = false;
                let mut saw_item_after_dropped = false;
                while let Some(frame) = anchor.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => {
                            if saw_dropped { saw_item_after_dropped = true; }
                            items.push(v);
                        }
                        Err(velo_streaming::StreamError::SenderDropped) => {
                            saw_dropped = true;
                            break; // Dropped is terminal; stream yields None next
                        }
                        other => panic!("unexpected frame: {:?}", other),
                    }
                }
                // Wait for sender task to complete
                send_task.await.expect("send task must not panic");
                assert_eq!(items.len(), 1000, "must receive exactly 1000 items");
                assert!(saw_dropped, "Dropped sentinel must be present");
                assert!(!saw_item_after_dropped, "no Item frame may follow the Dropped sentinel");
                assert_eq!(items, (0u32..1000).collect::<Vec<_>>(), "items must be in order");
                assert!(anchor.next().await.is_none(), "stream exhausted after Dropped");
            }

            /// TEST-11: Full attach/stream/finalize cycle against MockFrameTransport.
            ///
            /// Validates that the AnchorManager state machine correctly:
            /// - Creates an anchor with a typed StreamAnchor
            /// - Attaches a StreamSender
            /// - Delivers all sent items to the stream
            /// - Closes the stream after finalize()
            #[tokio::test(flavor = "multi_thread")]
            async fn test_11_mock_transport_full_cycle() {
                let mgr = manager().await;
                let mut anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let sender = mgr
                    .attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("attach must succeed");

                for i in 0u32..5 {
                    sender.send(i).await.expect("send");
                }
                sender.finalize().expect("finalize");

                let mut items = Vec::new();
                while let Some(frame) = anchor.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => items.push(v),
                        Ok(StreamFrame::Finalized) => break,
                        other => panic!("unexpected frame: {:?}", other),
                    }
                }
                assert_eq!(items, vec![0u32, 1, 2, 3, 4]);
                assert!(
                    anchor.next().await.is_none(),
                    "stream must yield None after Finalized"
                );
            }

            /// TEST-13: Unit coverage — handle roundtrip, registry lifecycle, exclusive attach.
            ///
            /// Validates:
            /// - StreamAnchorHandle::pack/unpack roundtrip preserves WorkerId and local_id
            /// - AnchorManager::create_anchor assigns monotonically increasing local IDs
            /// - A second concurrent attach returns AttachError::AlreadyAttached
            #[tokio::test(flavor = "multi_thread")]
            async fn test_13_unit_coverage() {
                // StreamAnchorHandle pack/unpack roundtrip
                let wid = WorkerId::from_u64(42);
                let handle = StreamAnchorHandle::pack(wid, 99);
                let (got_wid, got_lid) = handle.unpack();
                assert_eq!(got_wid, wid);
                assert_eq!(got_lid, 99u64);

                // Monotonically increasing local IDs
                let mgr = manager().await;
                let a1 = mgr.create_anchor::<u32>();
                let a2 = mgr.create_anchor::<u32>();
                let (_, lid1) = a1.handle().unpack();
                let (_, lid2) = a2.handle().unpack();
                assert!(lid2 > lid1, "local IDs must be monotonically increasing");

                // Exclusive attachment: second attach must be rejected
                let anchor = mgr.create_anchor::<u32>();
                let handle = anchor.handle();
                let _sender = mgr
                    .attach_stream_anchor::<u32>(handle)
                    .await
                    .expect("first attach must succeed");
                let result = mgr.attach_stream_anchor::<u32>(handle).await;
                assert!(
                    matches!(result, Err(AttachError::AlreadyAttached { .. })),
                    "second concurrent attach must return AlreadyAttached"
                );
            }
        }
    };
}
