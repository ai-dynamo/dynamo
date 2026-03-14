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
/// dropped because `StreamSender` writes frames directly to `AnchorStream` via the internal
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
            // StreamSender writes to frame_tx (the AnchorStream's internal channel),
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

/// Expand to a test module containing TEST-11 and TEST-13 integration scenarios.
///
/// # Parameters
/// - `$mod_name`: identifier for the generated module (e.g. `mock_suite`)
/// - `$make_manager`: expression that evaluates to `Arc<AnchorManager>`
///
/// # Tests generated
/// - `test_11_mock_transport_full_cycle`: create anchor, attach, send 5 items, finalize, collect
/// - `test_13_unit_coverage`: handle pack/unpack roundtrip, monotonic IDs, exclusive attach
#[macro_export]
macro_rules! run_transport_tests {
    ($mod_name:ident, $make_manager:expr) => {
        mod $mod_name {
            use super::*;
            use futures::StreamExt;
            use std::sync::Arc;
            use velo_common::WorkerId;
            use velo_streaming::{AnchorManager, AttachError, StreamAnchorHandle, StreamFrame};

            async fn manager() -> Arc<AnchorManager> {
                $make_manager
            }

            /// TEST-11: Full attach/stream/finalize cycle against MockFrameTransport.
            ///
            /// Validates that the AnchorManager state machine correctly:
            /// - Creates an anchor with a typed AnchorStream
            /// - Attaches a StreamSender
            /// - Delivers all sent items to the stream
            /// - Closes the stream after finalize()
            #[tokio::test(flavor = "multi_thread")]
            async fn test_11_mock_transport_full_cycle() {
                let mgr = manager().await;
                let (handle, mut stream) = mgr.create_anchor::<u32>();
                let sender = mgr
                    .attach_stream_anchor::<u32>(handle, "mock://1", 1)
                    .await
                    .expect("attach must succeed");

                for i in 0u32..5 {
                    sender.send(i).await.expect("send");
                }
                sender.finalize().expect("finalize");

                let mut items = Vec::new();
                while let Some(frame) = stream.next().await {
                    match frame {
                        Ok(StreamFrame::Item(v)) => items.push(v),
                        Ok(StreamFrame::Finalized) => break,
                        other => panic!("unexpected frame: {:?}", other),
                    }
                }
                assert_eq!(items, vec![0u32, 1, 2, 3, 4]);
                assert!(
                    stream.next().await.is_none(),
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
                let (h1, _s1) = mgr.create_anchor::<u32>();
                let (h2, _s2) = mgr.create_anchor::<u32>();
                let (_, lid1) = h1.unpack();
                let (_, lid2) = h2.unpack();
                assert!(lid2 > lid1, "local IDs must be monotonically increasing");

                // Exclusive attachment: second attach must be rejected
                let (handle, _stream) = mgr.create_anchor::<u32>();
                let _sender = mgr
                    .attach_stream_anchor::<u32>(handle, "mock://1", 1)
                    .await
                    .expect("first attach must succeed");
                let result = mgr.attach_stream_anchor::<u32>(handle, "mock://1", 2).await;
                assert!(
                    matches!(result, Err(AttachError::AlreadyAttached { .. })),
                    "second concurrent attach must return AlreadyAttached"
                );
            }
        }
    };
}
