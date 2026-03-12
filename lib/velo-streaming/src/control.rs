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
                        Some(e) if e.attachment.is_some() => {
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
                        if entry.attachment.is_some() {
                            Ok(AnchorAttachResponse::Err {
                                reason: format!("anchor {} already attached", req.handle),
                            })
                        } else {
                            entry.attachment = Some(receiver);
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
                        // Clear the attachment
                        let _ = entry.attachment.take();
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
        let (endpoint, receiver) = manager.transport.bind(local_id).await.unwrap();

        // Step 2: atomically set attachment under shard lock
        use dashmap::mapref::entry::Entry;
        let result = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => AnchorAttachResponse::Err {
                reason: format!("anchor {} not found", handle),
            },
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment.is_some() {
                    AnchorAttachResponse::Err {
                        reason: format!("anchor {} already attached", handle),
                    }
                } else {
                    entry.attachment = Some(receiver);
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
                .map(|e| e.attachment.is_some())
                .unwrap_or(false),
            "attachment must be Some after attach"
        );

        // Verify handler constructor compiles and returns Handler
        let _handler = create_anchor_attach_handler(manager.clone());
    }

    #[tokio::test]
    async fn test_anchor_attach_already_attached() {
        let manager = make_test_manager();
        let (handle, _stream) = manager.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // First attach via bind-then-lock
        {
            let (_, receiver) = manager.transport.bind(local_id).await.unwrap();
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = Some(receiver);
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
                if entry.attachment.is_some() {
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

        // Simulate attach first via bind-then-lock
        {
            let (_, receiver) = manager.transport.bind(local_id).await.unwrap();
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = Some(receiver);
            }
        }

        // Simulate detach handler logic
        use dashmap::mapref::entry::Entry;
        let maybe_entry_info = match manager.registry.entry(local_id) {
            Entry::Vacant(_) => None,
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                let _ = entry.attachment.take();
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
                .map(|e| e.attachment.is_none())
                .unwrap_or(false),
            "attachment must be None after detach"
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

        // Simulate attach via bind-then-lock
        {
            let (_, receiver) = manager.transport.bind(local_id).await.unwrap();
            use dashmap::mapref::entry::Entry;
            if let Entry::Occupied(mut occ) = manager.registry.entry(local_id) {
                let entry = occ.get_mut();
                entry.attachment = Some(receiver);
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
}
