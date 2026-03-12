// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control-plane handler constructors for the anchor lifecycle.
//!
//! This module provides four [`velo_messenger::Handler`] constructors:
//! - [`create_anchor_attach_handler`]: validates anchor existence, calls
//!   `transport.bind().await` (outside shard lock), then atomically stores
//!   the [`flume::Receiver`] in [`crate::anchor::AnchorEntry`].
//! - [`create_anchor_detach_handler`]: clears attachment, cancels CancellationToken,
//!   injects [`crate::frame::StreamFrame::Detached`] sentinel; anchor stays in registry.
//! - [`create_anchor_finalize_handler`]: injects [`crate::frame::StreamFrame::Finalized`]
//!   sentinel, then removes anchor from registry.
//! - [`create_anchor_cancel_handler`]: removes anchor from registry with no sentinel injection.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::anchor::AnchorManager;
use crate::handle::StreamAnchorHandle;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Request to attach a transport sender to an anchor.
///
/// `session_id` is an opaque caller-assigned identifier that may be forwarded
/// to the transport layer for logging and routing purposes.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnchorAttachRequest {
    pub handle: StreamAnchorHandle,
    pub session_id: u64,
}

/// Response from the attach handler.
#[derive(Debug, Serialize, Deserialize)]
pub enum AnchorAttachResponse {
    /// Attach succeeded; caller can connect to `stream_endpoint`.
    Ok { stream_endpoint: String },
    /// Attach failed; `reason` describes why.
    Err { reason: String },
}

/// Request to detach the current sender from an anchor without closing it.
///
/// After detach the anchor remains in the registry so a new sender may attach.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnchorDetachRequest {
    pub handle: StreamAnchorHandle,
}

/// Request to finalize (permanently close) an anchor.
///
/// After finalize the anchor is removed from the registry.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnchorFinalizeRequest {
    pub handle: StreamAnchorHandle,
}

/// Request to cancel an anchor with no sentinel injection.
///
/// Used when a sender exits before attaching or when an explicit abort is needed.
/// After cancel the anchor is removed from the registry.
#[derive(Debug, Serialize, Deserialize)]
pub struct AnchorCancelRequest {
    pub handle: StreamAnchorHandle,
}

// ---------------------------------------------------------------------------
// Reader pump
// ---------------------------------------------------------------------------

/// Reader pump: bridges transport frames to the anchor's delivery channel.
///
/// Spawned as a tokio task after successful attach. Reads from the transport
/// receiver, forwards to the anchor's frame_tx. Monitors for heartbeat
/// timeouts: 3 consecutive 5-second windows with no frames trigger Dropped
/// sentinel injection, registry removal (LIVE-02), and cleanup.
pub(crate) async fn reader_pump(
    transport_rx: flume::Receiver<Vec<u8>>,
    frame_tx: flume::Sender<Vec<u8>>,
    cancel_token: tokio_util::sync::CancellationToken,
    registry: std::sync::Arc<dashmap::DashMap<u64, crate::anchor::AnchorEntry>>,
    local_id: u64,
) {
    let mut missed_heartbeats: u8 = 0;
    let heartbeat_deadline = std::time::Duration::from_secs(5);

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = tokio::time::timeout(heartbeat_deadline, transport_rx.recv_async()) => {
                match result {
                    Ok(Ok(bytes)) => {
                        // Any frame (data or heartbeat) proves liveness
                        missed_heartbeats = 0;
                        // Forward to anchor's frame channel
                        if frame_tx.send_async(bytes).await.is_err() {
                            break; // consumer dropped
                        }
                    }
                    Ok(Err(_)) => break, // transport channel closed
                    Err(_timeout) => {
                        missed_heartbeats += 1;
                        if missed_heartbeats >= 3 {
                            // Inject Dropped sentinel -- sender is dead
                            if let Ok(dropped_bytes) = rmp_serde::to_vec(
                                &crate::frame::StreamFrame::<()>::Dropped
                            ) {
                                let _ = frame_tx.try_send(dropped_bytes);
                            }
                            // LIVE-02: Full anchor cleanup -- remove from registry
                            // so no stale entry remains (ANCR-04)
                            if let Some((_, entry)) = registry.remove(&local_id) {
                                entry.cancel_token.cancel();
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    // Cleanup: cancel token so other paths know the pump exited
    cancel_token.cancel();
}

// ---------------------------------------------------------------------------
// Handler constructors
// ---------------------------------------------------------------------------

/// Build the `_anchor_attach` handler.
///
/// Uses the bind-then-lock pattern: calls `transport.bind().await` OUTSIDE the
/// DashMap shard lock, then atomically checks and sets the attachment under the lock.
/// This avoids holding the shard lock across an async `.await` point.
///
/// Returns [`AnchorAttachResponse::Ok`] on success or [`AnchorAttachResponse::Err`] on
/// any failure (not found, already attached, transport error).
pub fn create_anchor_attach_handler(manager: Arc<AnchorManager>) -> velo_messenger::Handler {
    velo_messenger::Handler::typed_unary_async(
        "_anchor_attach",
        move |ctx: velo_messenger::TypedContext<AnchorAttachRequest>| {
            let manager = manager.clone();
            async move {
                let req = ctx.input;
                let (_, local_id) = req.handle.unpack();

                // Step 1: Quick check -- anchor exists and is unattached (drop lock)
                {
                    let entry = manager.registry.get(&local_id);
                    match entry {
                        None => {
                            return Ok(AnchorAttachResponse::Err {
                                reason: format!("anchor {} not found", req.handle),
                            });
                        }
                        Some(e) if e.attachment => {
                            return Ok(AnchorAttachResponse::Err {
                                reason: format!("anchor {} already attached", req.handle),
                            });
                        }
                        _ => {} // looks good, proceed
                    }
                } // DashMap ref dropped here

                // Step 2: Async bind OUTSIDE shard lock
                let (endpoint, receiver) = match manager.transport.bind(local_id).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        return Ok(AnchorAttachResponse::Err {
                            reason: format!("transport error: {}", e),
                        });
                    }
                };

                // Step 3: Atomically set attachment under shard lock
                use dashmap::mapref::entry::Entry;
                match manager.registry.entry(local_id) {
                    Entry::Vacant(_) => Ok(AnchorAttachResponse::Err {
                        reason: format!("anchor {} removed during bind", req.handle),
                    }),
                    Entry::Occupied(mut occ) => {
                        let entry = occ.get_mut();
                        if entry.attachment {
                            Ok(AnchorAttachResponse::Err {
                                reason: format!("anchor {} already attached", req.handle),
                            })
                        } else {
                            // Clone frame_tx and cancel_token for reader pump
                            let pump_frame_tx = entry.frame_tx.clone();
                            let pump_cancel = entry.cancel_token.clone();

                            // Mark as attached
                            entry.attachment = true;

                            // Drop shard lock before spawning
                            drop(occ);

                            // Spawn reader pump as background task
                            let pump_registry = manager.registry.clone();
                            let (_, local_id) = req.handle.unpack();
                            tokio::spawn(reader_pump(
                                receiver,        // transport receiver from bind
                                pump_frame_tx,   // cloned from entry
                                pump_cancel,     // cloned from entry
                                pump_registry,   // Arc clone of registry
                                local_id,        // anchor's local_id
                            ));

                            Ok(AnchorAttachResponse::Ok {
                                stream_endpoint: endpoint,
                            })
                        }
                    }
                }
            }
        },
    )
    .spawn()
    .build()
}

/// Build the `_anchor_detach` handler.
///
/// Atomically clears `attachment` via `DashMap::entry()`, then -- after dropping the
/// shard lock -- cancels the `CancellationToken` and injects a
/// [`crate::frame::StreamFrame::Detached`] sentinel into the frame channel.
/// The anchor remains in the registry so a new sender may re-attach.
///
/// Idempotent: if the anchor is not found, returns `Ok(())`.
pub fn create_anchor_detach_handler(manager: Arc<AnchorManager>) -> velo_messenger::Handler {
    velo_messenger::Handler::typed_unary_async(
        "_anchor_detach",
        move |ctx: velo_messenger::TypedContext<AnchorDetachRequest>| {
            let manager = manager.clone();
            async move {
                let req = ctx.input;
                let (_, local_id) = req.handle.unpack();

                use dashmap::mapref::entry::Entry;
                // Atomically clear attachment and clone cancel_token + frame_tx
                // before dropping the shard lock (never hold DashMap ref across channel ops).
                let maybe_entry_info = match manager.registry.entry(local_id) {
                    Entry::Vacant(_) => None,
                    Entry::Occupied(mut occ) => {
                        let entry = occ.get_mut();
                        // Clear the attachment flag
                        entry.attachment = false;
                        Some((entry.cancel_token.clone(), entry.frame_tx.clone()))
                    }
                };
                // shard lock is now dropped

                if let Some((cancel_token, frame_tx)) = maybe_entry_info {
                    cancel_token.cancel();
                    let sentinel_bytes = rmp_serde::to_vec(
                        &crate::frame::StreamFrame::<Vec<u8>>::Detached,
                    )
                    .expect("serialize Detached sentinel");
                    let _ = frame_tx.try_send(sentinel_bytes);
                }

                Ok(())
            }
        },
    )
    .spawn()
    .build()
}

/// Build the `_anchor_finalize` handler.
///
/// Atomically removes the anchor from the registry via `remove_anchor()`, injects a
/// [`crate::frame::StreamFrame::Finalized`] sentinel, and cancels the `CancellationToken`.
///
/// Idempotent: if the anchor is already absent, returns `Ok(())`.
pub fn create_anchor_finalize_handler(manager: Arc<AnchorManager>) -> velo_messenger::Handler {
    velo_messenger::Handler::typed_unary_async(
        "_anchor_finalize",
        move |ctx: velo_messenger::TypedContext<AnchorFinalizeRequest>| {
            let manager = manager.clone();
            async move {
                let req = ctx.input;
                let (_, local_id) = req.handle.unpack();

                // remove_anchor cancels the token and returns the entry
                if let Some(entry) = manager.remove_anchor(local_id) {
                    let sentinel_bytes = rmp_serde::to_vec(
                        &crate::frame::StreamFrame::<Vec<u8>>::Finalized,
                    )
                    .expect("serialize Finalized sentinel");
                    let _ = entry.frame_tx.try_send(sentinel_bytes);
                }

                Ok(())
            }
        },
    )
    .spawn()
    .build()
}

/// Build the `_anchor_cancel` handler.
///
/// Removes the anchor from the registry with no sentinel injection.
/// Used when a sender aborts before or during attachment.
///
/// Idempotent: calling cancel on an already-absent anchor does not panic.
pub fn create_anchor_cancel_handler(manager: Arc<AnchorManager>) -> velo_messenger::Handler {
    velo_messenger::Handler::typed_unary_async(
        "_anchor_cancel",
        move |ctx: velo_messenger::TypedContext<AnchorCancelRequest>| {
            let manager = manager.clone();
            async move {
                let req = ctx.input;
                let (_, local_id) = req.handle.unpack();

                // remove_anchor is a no-op (returns None) if anchor absent -- idempotent
                if let Some(entry) = manager.remove_anchor(local_id) {
                    entry.cancel_token.cancel();
                }

                Ok(())
            }
        },
    )
    .spawn()
    .build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyhowResult;
    use futures::future::BoxFuture;
    use futures::StreamExt;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // MockFrameTransport (test-only)
    // -----------------------------------------------------------------------

    struct MockFrameTransport;

    impl crate::transport::FrameTransport for MockFrameTransport {
        fn bind(&self, _anchor_id: u64) -> BoxFuture<'_, AnyhowResult<(String, flume::Receiver<Vec<u8>>)>> {
            Box::pin(async {
                Ok(("mock://test-endpoint".to_string(), flume::bounded::<Vec<u8>>(256).1))
            })
        }

        fn connect(
            &self,
            _endpoint: &str,
            _anchor_id: u64,
            _session_id: u64,
        ) -> BoxFuture<'_, AnyhowResult<flume::Sender<Vec<u8>>>> {
            Box::pin(async {
                Ok(flume::bounded::<Vec<u8>>(256).0)
            })
        }
    }

    // -----------------------------------------------------------------------
    // Helper: make a test AnchorManager
    // -----------------------------------------------------------------------

    fn make_test_manager() -> Arc<AnchorManager> {
        let worker_id = velo_common::WorkerId::from_u64(1);
        let transport = Arc::new(MockFrameTransport);
        Arc::new(AnchorManager::new(worker_id, transport))
    }

    // -----------------------------------------------------------------------
    // Test helpers for calling handler logic directly
    // -----------------------------------------------------------------------

    // We call the handler constructor only to verify it compiles and returns Handler.
    // For behavioral tests, we call the underlying AnchorManager APIs + simulate
    // the same logic the handler performs to verify correctness without needing
    // a running velo_messenger runtime.

    // -----------------------------------------------------------------------
    // Type serialization tests (Task 1 scope)
    // -----------------------------------------------------------------------

    #[test]
    fn test_anchor_attach_response_serde_ok() {
        let resp = AnchorAttachResponse::Ok {
            stream_endpoint: "mock://test-endpoint".to_string(),
        };
        let json = serde_json::to_string(&resp).expect("serialize Ok");
        let decoded: AnchorAttachResponse = serde_json::from_str(&json).expect("deserialize Ok");
        match decoded {
            AnchorAttachResponse::Ok { stream_endpoint } => {
                assert_eq!(stream_endpoint, "mock://test-endpoint");
            }
            other => panic!("expected Ok, got {:?}", other),
        }
    }

    #[test]
    fn test_anchor_attach_response_serde_err() {
        let resp = AnchorAttachResponse::Err {
            reason: "already attached".to_string(),
        };
        let json = serde_json::to_string(&resp).expect("serialize Err");
        let decoded: AnchorAttachResponse = serde_json::from_str(&json).expect("deserialize Err");
        match decoded {
            AnchorAttachResponse::Err { reason } => {
                assert!(reason.contains("already attached"));
            }
            other => panic!("expected Err, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Attach handler tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_anchor_attach_handler() {
        let manager = make_test_manager();
        let (handle, _stream) = manager.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // Simulate bind-then-lock attach handler logic:
        // Step 1: async bind outside shard lock
        let (endpoint, _receiver) = manager.transport.bind(local_id).await.unwrap();

        // Step 2: atomically set attachment under shard lock
        use dashmap::mapref::entry::Entry;
        let result = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => AnchorAttachResponse::Err {
                reason: format!("anchor {} not found", handle),
            },
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment {
                    AnchorAttachResponse::Err {
                        reason: format!("anchor {} already attached", handle),
                    }
                } else {
                    entry.attachment = true;
                    AnchorAttachResponse::Ok {
                        stream_endpoint: endpoint,
                    }
                }
            }
        };

        match result {
            AnchorAttachResponse::Ok { stream_endpoint } => {
                assert_eq!(stream_endpoint, "mock://test-endpoint");
            }
            other => panic!("expected Ok, got {:?}", other),
        }

        // Verify attachment is set
        assert!(
            manager
                .registry
                .get(&local_id)
                .map(|e| e.attachment)
                .unwrap_or(false),
            "attachment must be true after attach"
        );

        // Verify handler constructor compiles and returns Handler
        let _handler = create_anchor_attach_handler(manager.clone());
    }

    #[tokio::test]
    async fn test_anchor_attach_already_attached() {
        let manager = make_test_manager();
        let (handle, _stream) = manager.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // First attach: set attachment flag directly
        {
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = true;
            }
        }

        // Second attach via handler logic -- should fail
        use dashmap::mapref::entry::Entry;
        let result = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => AnchorAttachResponse::Err {
                reason: format!("anchor {} not found", handle),
            },
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment {
                    AnchorAttachResponse::Err {
                        reason: format!("anchor {} already attached", handle),
                    }
                } else {
                    AnchorAttachResponse::Ok {
                        stream_endpoint: "unreachable".to_string(),
                    }
                }
            }
        };

        match result {
            AnchorAttachResponse::Err { reason } => {
                assert!(
                    reason.contains("already attached"),
                    "reason must mention 'already attached', got: {reason}"
                );
            }
            other => panic!("expected Err, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_anchor_attach_not_found() {
        let manager = make_test_manager();
        // Create a handle that is NOT in the registry
        let fake_handle =
            StreamAnchorHandle::pack(velo_common::WorkerId::from_u64(1), 9999);

        // Simulate handler logic
        use dashmap::mapref::entry::Entry;
        let local_id = 9999u64;
        let result = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => AnchorAttachResponse::Err {
                reason: format!("anchor {} not found", fake_handle),
            },
            Entry::Occupied(_) => panic!("should not be occupied"),
        };

        match result {
            AnchorAttachResponse::Err { reason } => {
                assert!(
                    reason.contains("not found"),
                    "reason must mention 'not found', got: {reason}"
                );
            }
            other => panic!("expected Err, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Detach handler tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_anchor_detach_handler() {
        let manager = make_test_manager();
        let (handle, mut stream) = manager.create_anchor::<Vec<u8>>();
        let (_, local_id) = handle.unpack();

        // Simulate attach: set flag directly
        {
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = true;
            }
        }

        // Simulate detach handler logic
        use dashmap::mapref::entry::Entry;
        let maybe_entry_info = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => None,
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                entry.attachment = false;
                Some((entry.cancel_token.clone(), entry.frame_tx.clone()))
            }
        };

        if let Some((cancel_token, frame_tx)) = maybe_entry_info {
            cancel_token.cancel();
            let sentinel_bytes = rmp_serde::to_vec(
                &crate::frame::StreamFrame::<Vec<u8>>::Detached,
            )
            .expect("serialize Detached sentinel");
            let _ = frame_tx.try_send(sentinel_bytes);
        }

        // Verify: attachment is cleared
        assert!(
            manager
                .registry
                .get(&local_id)
                .map(|e| !e.attachment)
                .unwrap_or(false),
            "attachment must be false after detach"
        );

        // Verify: anchor still in registry
        assert!(
            manager.registry.contains_key(&local_id),
            "anchor must remain in registry after detach"
        );

        // Verify: Detached sentinel received via Stream interface
        let result = stream.next().await;
        assert!(
            matches!(result, Some(Ok(crate::frame::StreamFrame::Detached))),
            "sentinel must be Detached, got {:?}",
            result
        );

        // Verify handler constructor compiles
        let _handler = create_anchor_detach_handler(manager.clone());
    }

    // -----------------------------------------------------------------------
    // Finalize handler tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_anchor_finalize_handler() {
        let manager = make_test_manager();
        let (handle, mut stream) = manager.create_anchor::<Vec<u8>>();
        let (_, local_id) = handle.unpack();

        // Simulate attach: set flag directly
        {
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = true;
            }
        }

        // Simulate finalize handler logic
        if let Some(entry) = manager.remove_anchor(local_id) {
            let sentinel_bytes = rmp_serde::to_vec(
                &crate::frame::StreamFrame::<Vec<u8>>::Finalized,
            )
            .expect("serialize Finalized sentinel");
            let _ = entry.frame_tx.try_send(sentinel_bytes);
        }

        // Verify: anchor removed from registry
        assert!(
            !manager.registry.contains_key(&local_id),
            "anchor must be absent from registry after finalize"
        );

        // Verify: Finalized sentinel received via Stream interface
        let result = stream.next().await;
        assert!(
            matches!(result, Some(Ok(crate::frame::StreamFrame::Finalized))),
            "sentinel must be Finalized, got {:?}",
            result
        );

        // Verify handler constructor compiles
        let _handler = create_anchor_finalize_handler(manager.clone());
    }

    // -----------------------------------------------------------------------
    // Cancel handler tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_anchor_cancel_handler() {
        let manager = make_test_manager();
        let (handle, _stream) = manager.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // Simulate cancel handler logic
        if let Some(entry) = manager.remove_anchor(local_id) {
            entry.cancel_token.cancel();
        }

        // Verify: anchor removed
        assert!(
            !manager.registry.contains_key(&local_id),
            "anchor must be absent after cancel"
        );

        // Idempotent: cancel again -- must not panic
        if let Some(entry) = manager.remove_anchor(local_id) {
            entry.cancel_token.cancel();
        }
        // No panic -- test passes

        // Verify handler constructor compiles
        let _handler = create_anchor_cancel_handler(manager.clone());
    }

    // -----------------------------------------------------------------------
    // reader_pump tests (Plan 08-03, Task 2)
    // -----------------------------------------------------------------------

    /// Helper: set up infrastructure for reader_pump tests.
    /// Returns (transport_tx, frame_rx, cancel_token, registry, local_id).
    fn make_pump_test_infra() -> (
        flume::Sender<Vec<u8>>,     // transport_tx: simulates transport frames
        flume::Receiver<Vec<u8>>,   // frame_rx: where pump writes to (consumer side)
        tokio_util::sync::CancellationToken,
        std::sync::Arc<dashmap::DashMap<u64, crate::anchor::AnchorEntry>>,
        u64,                         // local_id
    ) {
        let (transport_tx, transport_rx) = flume::bounded::<Vec<u8>>(256);
        let (frame_tx, frame_rx) = flume::bounded::<Vec<u8>>(256);
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let registry = std::sync::Arc::new(dashmap::DashMap::new());
        let local_id = 1u64;

        // Insert an entry in the registry so the pump can remove it
        registry.insert(local_id, crate::anchor::AnchorEntry {
            frame_tx: frame_tx.clone(),
            cancel_token: cancel_token.clone(),
            attachment: true,
        });

        // Spawn the reader pump
        let pump_cancel = cancel_token.clone();
        let pump_registry = registry.clone();
        tokio::spawn(reader_pump(
            transport_rx,
            frame_tx,
            pump_cancel,
            pump_registry,
            local_id,
        ));

        (transport_tx, frame_rx, cancel_token, registry, local_id)
    }

    #[tokio::test]
    async fn test_pump_forwards_data_frames() {
        let (transport_tx, frame_rx, _cancel, _registry, _id) = make_pump_test_infra();

        // Send a data frame through the transport side
        let data_bytes = rmp_serde::to_vec(&crate::frame::StreamFrame::Item(42u32)).unwrap();
        transport_tx.send_async(data_bytes.clone()).await.unwrap();

        // Should arrive on the frame_rx side
        let received = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            frame_rx.recv_async(),
        ).await.expect("timeout waiting for frame").expect("frame_rx closed");

        assert_eq!(received, data_bytes, "pump must forward bytes unchanged");
    }

    #[tokio::test]
    async fn test_pump_resets_heartbeat_counter_on_frame() {
        tokio::time::pause();

        let (transport_tx, frame_rx, _cancel, registry, local_id) = make_pump_test_infra();

        // Wait 4.5 seconds (almost one heartbeat window)
        tokio::time::sleep(std::time::Duration::from_millis(4500)).await;

        // Send a frame to reset the counter
        let hb_bytes = rmp_serde::to_vec(&crate::frame::StreamFrame::<()>::Heartbeat).unwrap();
        transport_tx.send_async(hb_bytes).await.unwrap();

        // Wait another 4.5 seconds
        tokio::time::sleep(std::time::Duration::from_millis(4500)).await;

        // Send another frame
        transport_tx.send_async(
            rmp_serde::to_vec(&crate::frame::StreamFrame::<()>::Heartbeat).unwrap()
        ).await.unwrap();

        // Drain forwarded frames
        while frame_rx.try_recv().is_ok() {}

        // The anchor should still be in the registry (counter resets each time)
        assert!(
            registry.contains_key(&local_id),
            "anchor must still be in registry -- heartbeat counter was reset"
        );
    }

    #[tokio::test]
    async fn test_pump_injects_dropped_after_3_missed_heartbeats() {
        tokio::time::pause();

        let (transport_tx, frame_rx, _cancel, _registry, _id) = make_pump_test_infra();

        // Keep transport_tx alive but don't send anything -- pump will timeout
        // 3 consecutive 5s windows with no frames trigger Dropped
        tokio::time::sleep(std::time::Duration::from_secs(16)).await;

        // Collect all frames from frame_rx
        let mut frames = Vec::new();
        while let Ok(bytes) = frame_rx.try_recv() {
            frames.push(bytes);
        }

        // The last frame should be a Dropped sentinel
        assert!(!frames.is_empty(), "must have received at least one frame (Dropped sentinel)");
        let last = frames.last().unwrap();
        let decoded: crate::frame::StreamFrame<()> = rmp_serde::from_slice(last).expect("deserialize");
        assert!(
            matches!(decoded, crate::frame::StreamFrame::Dropped),
            "last frame must be Dropped, got {:?}",
            decoded
        );

        // Keep transport_tx alive for the duration of the test
        drop(transport_tx);
    }

    #[tokio::test]
    async fn test_pump_removes_registry_entry_after_3_missed_heartbeats() {
        tokio::time::pause();

        let (transport_tx, _frame_rx, _cancel, registry, local_id) = make_pump_test_infra();

        // Keep transport_tx alive but don't send -- pump will timeout
        tokio::time::sleep(std::time::Duration::from_secs(16)).await;

        // LIVE-02: anchor entry must be removed from registry
        assert!(
            !registry.contains_key(&local_id),
            "anchor must be removed from registry after 3 missed heartbeats (LIVE-02)"
        );

        // Keep transport_tx alive for the duration of the test
        drop(transport_tx);
    }

    #[tokio::test]
    async fn test_pump_exits_when_cancel_token_cancelled() {
        let (transport_tx, frame_rx, cancel_token, registry, local_id) = make_pump_test_infra();

        // Cancel the token
        cancel_token.cancel();

        // Give the pump a moment to exit
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pump should have exited -- sending on transport_tx should not be forwarded
        let data = rmp_serde::to_vec(&crate::frame::StreamFrame::Item(99u32)).unwrap();
        let _ = transport_tx.try_send(data);

        // Allow propagation
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // frame_rx should be empty (pump exited, nothing forwarded)
        assert!(
            frame_rx.try_recv().is_err(),
            "no frames should be forwarded after cancel"
        );

        // Pump calls cancel on exit, so token should be cancelled
        assert!(cancel_token.is_cancelled());

        // Registry entry may or may not be removed (cancel != heartbeat death)
        let _ = (registry, local_id);
    }

    #[tokio::test]
    async fn test_pump_exits_when_transport_closes() {
        let (transport_tx, _frame_rx, cancel_token, _registry, _id) = make_pump_test_infra();

        // Drop the transport sender -- transport channel closes
        drop(transport_tx);

        // Give the pump a moment to exit
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pump should have exited and cancelled the token
        assert!(
            cancel_token.is_cancelled(),
            "cancel_token must be cancelled after pump exits due to transport close"
        );
    }

    #[tokio::test]
    async fn test_pump_exits_when_consumer_drops() {
        let (transport_tx, frame_rx, cancel_token, _registry, _id) = make_pump_test_infra();

        // Drop the frame_rx consumer side -- pump's send will fail
        drop(frame_rx);

        // Send data so the pump tries to forward and fails
        let data = rmp_serde::to_vec(&crate::frame::StreamFrame::Item(1u32)).unwrap();
        let _ = transport_tx.send_async(data).await;

        // Give the pump time to process and exit
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pump should have exited and cancelled the token
        assert!(
            cancel_token.is_cancelled(),
            "cancel_token must be cancelled after pump exits due to consumer drop"
        );
    }
}
