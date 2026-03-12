// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anchor registry layer: [`AnchorManager`], [`AnchorEntry`], [`AnchorStream`], and [`AttachError`].
//!
//! The anchor registry is the core coordination point for the streaming protocol.
//! Each anchor represents a single exclusive-attachment stream slot:
//!
//! - [`AnchorManager::create_anchor`] allocates a registry slot and returns a
//!   [`crate::handle::StreamAnchorHandle`] (for use by control handlers) and an
//!   [`AnchorStream<T>`] (for the consumer).
//! - Exactly one [`flume::Sender`] may be attached at a time;
//!   the attach check is performed atomically via [`dashmap::DashMap::entry`].
//! - Each entry holds a [`tokio_util::sync::CancellationToken`] created at anchor
//!   creation so that whichever cleanup path fires first cancels the token; subsequent
//!   cancellations are no-ops.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use dashmap::DashMap;
use tokio_util::sync::CancellationToken;

use crate::handle::StreamAnchorHandle;

// ---------------------------------------------------------------------------
// AttachError
// ---------------------------------------------------------------------------

/// Errors that can occur when attempting to attach a sender to an anchor.
#[derive(Debug, thiserror::Error)]
pub enum AttachError {
    /// The requested anchor handle was not found in the registry.
    #[error("anchor {handle} not found in registry")]
    AnchorNotFound { handle: StreamAnchorHandle },

    /// Another sender is already attached to this anchor.
    #[error("anchor {handle} is already attached")]
    AlreadyAttached { handle: StreamAnchorHandle },

    /// The underlying transport failed during bind/connect.
    #[error("transport bind failed: {0}")]
    TransportError(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// AnchorEntry
// ---------------------------------------------------------------------------

/// A single slot in the anchor registry.
///
/// Non-generic by design: [`AnchorManager`] stores `DashMap<u64, AnchorEntry>`
/// which avoids propagating a type parameter throughout the registry.
///
/// The `attachment` field stores the exclusive [`flume::Receiver`] (the receive
/// half of the transport channel) while a sender is connected. The check-and-set
/// is performed atomically via [`dashmap::mapref::entry::Entry`] to prevent
/// TOCTOU races.
// Fields are consumed by Phase 7+ control handlers and Phase 8 data path.
#[allow(dead_code)]
pub(crate) struct AnchorEntry {
    /// Raw-bytes frame delivery channel to the [`AnchorStream<T>`] consumer.
    ///
    /// Non-generic so `DashMap<u64, AnchorEntry>` requires no type parameters.
    pub frame_tx: flume::Sender<Vec<u8>>,

    /// Created at anchor creation; cancelled by whichever cleanup path fires first.
    ///
    /// `cancel()` is idempotent -- a second caller is a no-op.
    pub cancel_token: CancellationToken,

    /// Present iff a sender is currently attached.
    ///
    /// Holds the receive half of the transport channel returned by
    /// [`crate::transport::FrameTransport::bind`].
    pub attachment: Option<flume::Receiver<Vec<u8>>>,
}

// ---------------------------------------------------------------------------
// AnchorStream<T>
// ---------------------------------------------------------------------------

/// Thin wrapper around the receive half of the per-anchor flume channel.
///
/// Phase 8 will add the async `next()` / `Stream` implementation that
/// deserializes the raw bytes into `StreamFrame<T>`. For now this is a
/// stub that holds the receiver and exposes it for testing.
pub struct AnchorStream<T> {
    // Consumed by Phase 8 Stream impl.
    #[allow(dead_code)]
    pub(crate) rx: flume::Receiver<Vec<u8>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> AnchorStream<T> {
    pub(crate) fn new(rx: flume::Receiver<Vec<u8>>) -> Self {
        Self {
            rx,
            _phantom: std::marker::PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// AnchorManager
// ---------------------------------------------------------------------------

/// Central registry that creates and tracks streaming anchors.
///
/// `worker_id` is stamped into every [`StreamAnchorHandle`] so that remote
/// peers can route responses back to the correct worker. `next_local_id`
/// starts at 0 and is incremented with `fetch_add(1)` -- the *result + 1*
/// is the first valid local ID (i.e., IDs start at 1; 0 is reserved).
///
/// The `registry` is wrapped in an `Arc` so that control handlers (Phase 7)
/// and the data-path pump (Phase 8) can hold a cheap clone of the registry
/// reference without holding a reference to the whole `AnchorManager`.
pub struct AnchorManager {
    worker_id: velo_common::WorkerId,
    next_local_id: AtomicU64,
    pub(crate) registry: Arc<DashMap<u64, AnchorEntry>>,
    pub transport: Arc<dyn crate::transport::FrameTransport>,
}

impl AnchorManager {
    /// Create a new [`AnchorManager`] with the given worker identity and transport.
    pub fn new(
        worker_id: velo_common::WorkerId,
        transport: Arc<dyn crate::transport::FrameTransport>,
    ) -> Self {
        Self {
            worker_id,
            next_local_id: AtomicU64::new(0),
            registry: Arc::new(DashMap::new()),
            transport,
        }
    }

    /// Allocate a new anchor and return the handle (for control path) and stream (for consumer).
    ///
    /// Local IDs start at 1 and increment monotonically; ID 0 is reserved.
    /// A flume bounded channel (capacity 256) is created per anchor to deliver raw frame bytes.
    pub fn create_anchor<T>(&self) -> (StreamAnchorHandle, AnchorStream<T>) {
        // fetch_add returns the *old* value (starts at 0), so +1 gives us IDs starting at 1.
        let local_id = self.next_local_id.fetch_add(1, Ordering::Relaxed) + 1;

        let (frame_tx, frame_rx) = flume::bounded::<Vec<u8>>(256);
        let cancel_token = CancellationToken::new();

        let entry = AnchorEntry {
            frame_tx,
            cancel_token,
            attachment: None,
        };

        self.registry.insert(local_id, entry);

        let handle = StreamAnchorHandle::pack(self.worker_id, local_id);
        let stream = AnchorStream::new(frame_rx);
        (handle, stream)
    }

    /// Remove an anchor from the registry and return its entry (if present).
    ///
    /// Cancels the entry's token before returning. Used by control path cleanup
    /// handlers (Phase 7) and drop impls.
    #[allow(dead_code)]
    pub(crate) fn remove_anchor(&self, local_id: u64) -> Option<AnchorEntry> {
        self.registry.remove(&local_id).map(|(_, entry)| {
            entry.cancel_token.cancel();
            entry
        })
    }

    /// Inject a raw sentinel frame into the anchor's delivery channel.
    ///
    /// This is a non-blocking best-effort send used by the control path (Phase 7).
    /// The data path (Phase 8) will use a blocking variant for `Item` frames.
    ///
    /// # Note
    /// The registry reference is dropped before any other operation to ensure we do
    /// NOT hold a DashMap shard lock across any await point.
    #[allow(dead_code)]
    pub(crate) fn inject_sentinel(&self, local_id: u64, frame_bytes: Vec<u8>) {
        // Obtain a cloned Sender so we drop the DashMap reference immediately.
        let maybe_sender = self
            .registry
            .get(&local_id)
            .map(|entry| entry.frame_tx.clone());

        if let Some(sender) = maybe_sender {
            // Non-blocking best-effort -- control sentinels must never stall.
            let _ = sender.try_send(frame_bytes);
        }
    }

    /// Atomically attempt to attach a [`flume::Receiver`] to an anchor.
    ///
    /// Uses `DashMap::entry()` to perform the check-and-set atomically under
    /// the shard lock, preventing TOCTOU races between concurrent attach attempts.
    ///
    /// Returns `Err(AttachError::AlreadyAttached)` if a reader is already attached.
    /// Returns `Err(AttachError::AnchorNotFound)` if `local_id` is not in the registry.
    #[allow(dead_code)]
    pub(crate) fn try_attach(
        &self,
        local_id: u64,
        reader: flume::Receiver<Vec<u8>>,
        handle: StreamAnchorHandle,
    ) -> Result<(), AttachError> {
        use dashmap::mapref::entry::Entry;
        match self.registry.entry(local_id) {
            Entry::Vacant(_) => Err(AttachError::AnchorNotFound { handle }),
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment.is_some() {
                    Err(AttachError::AlreadyAttached { handle })
                } else {
                    entry.attachment = Some(reader);
                    Ok(())
                }
            }
        }
    }

    /// Detach the current [`flume::Receiver`] from an anchor, returning it if present.
    #[allow(dead_code)]
    pub(crate) fn detach(&self, local_id: u64) -> Option<flume::Receiver<Vec<u8>>> {
        self.registry
            .get_mut(&local_id)
            .and_then(|mut entry| entry.attachment.take())
    }

    /// Attach a sender to an existing anchor, establishing the transport connection.
    ///
    /// This is the primary sender-side entry point (API-05). It:
    /// 1. Validates the anchor exists and is unattached
    /// 2. Calls `transport.connect()` to establish the write channel
    /// 3. Atomically marks the anchor as attached
    /// 4. Returns a [`StreamSender<T>`](crate::sender::StreamSender) for pushing typed frames
    ///
    /// The StreamSender writes to the entry's `frame_tx` so items flow directly
    /// to the [`AnchorStream<T>`] consumer. The transport connection validates
    /// network setup; the reader pump (Plan 03) will forward transport-received
    /// frames to `frame_tx` for cross-worker flows.
    ///
    /// # Errors
    /// - [`AttachError::AnchorNotFound`] if the handle is not in the registry
    /// - [`AttachError::AlreadyAttached`] if another sender is already connected
    /// - [`AttachError::TransportError`] if `transport.connect()` fails
    pub async fn attach_stream_anchor<T: serde::Serialize>(
        &self,
        handle: StreamAnchorHandle,
        endpoint: &str,
        session_id: u64,
    ) -> Result<crate::sender::StreamSender<T>, AttachError> {
        let (_, local_id) = handle.unpack();

        // Step 1: Quick check anchor exists and is unattached (drop ref before async)
        {
            let entry = self.registry.get(&local_id);
            match entry {
                None => return Err(AttachError::AnchorNotFound { handle }),
                Some(e) if e.attachment.is_some() => {
                    return Err(AttachError::AlreadyAttached { handle });
                }
                _ => {} // looks good, proceed
            }
        } // DashMap ref dropped here

        // Step 2: Async connect OUTSIDE shard lock — validates transport setup
        let _transport_tx = self
            .transport
            .connect(endpoint, local_id, session_id)
            .await?; // maps to AttachError::TransportError via From<anyhow::Error>

        // Step 3: Atomically set attachment under shard lock.
        // Re-check under the entry guard to prevent TOCTOU.
        use dashmap::mapref::entry::Entry;
        match self.registry.entry(local_id) {
            Entry::Vacant(_) => Err(AttachError::AnchorNotFound { handle }),
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment.is_some() {
                    Err(AttachError::AlreadyAttached { handle })
                } else {
                    // Clone the frame_tx so the StreamSender can write items
                    // directly to the AnchorStream consumer.
                    let frame_tx = entry.frame_tx.clone();

                    // Mark as attached with a dummy receiver (Plan 03 will
                    // change attachment to bool since the reader pump takes
                    // ownership of the transport receiver separately).
                    entry.attachment = Some(flume::bounded::<Vec<u8>>(1).1);

                    Ok(crate::sender::StreamSender::new(frame_tx, handle))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result as AnyhowResult;
    use futures::future::BoxFuture;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Mock transport for unit tests
    // -----------------------------------------------------------------------

    struct MockTransport;

    impl crate::transport::FrameTransport for MockTransport {
        fn bind(&self, _anchor_id: u64) -> BoxFuture<'_, AnyhowResult<(String, flume::Receiver<Vec<u8>>)>> {
            Box::pin(async {
                Ok(("mock://endpoint".to_string(), flume::bounded::<Vec<u8>>(256).1))
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

    fn make_manager() -> AnchorManager {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport = Arc::new(MockTransport);
        AnchorManager::new(worker_id, transport)
    }

    /// A configurable mock transport that can simulate connect failures.
    struct ConfigurableMockTransport {
        connect_should_fail: std::sync::atomic::AtomicBool,
    }

    impl ConfigurableMockTransport {
        fn new() -> Self {
            Self {
                connect_should_fail: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn set_connect_fail(&self, fail: bool) {
            self.connect_should_fail
                .store(fail, std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl crate::transport::FrameTransport for ConfigurableMockTransport {
        fn bind(
            &self,
            _anchor_id: u64,
        ) -> BoxFuture<'_, AnyhowResult<(String, flume::Receiver<Vec<u8>>)>> {
            Box::pin(async {
                Ok(("mock://endpoint".to_string(), flume::bounded::<Vec<u8>>(256).1))
            })
        }

        fn connect(
            &self,
            _endpoint: &str,
            _anchor_id: u64,
            _session_id: u64,
        ) -> BoxFuture<'_, AnyhowResult<flume::Sender<Vec<u8>>>> {
            let should_fail = self
                .connect_should_fail
                .load(std::sync::atomic::Ordering::Relaxed);
            Box::pin(async move {
                if should_fail {
                    Err(anyhow::anyhow!("mock transport connect failure"))
                } else {
                    Ok(flume::bounded::<Vec<u8>>(256).0)
                }
            })
        }
    }

    fn make_configurable_manager() -> (AnchorManager, Arc<ConfigurableMockTransport>) {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport = Arc::new(ConfigurableMockTransport::new());
        let mgr = AnchorManager::new(worker_id, transport.clone());
        (mgr, transport)
    }

    // -----------------------------------------------------------------------
    // Test 1: Monotonic local IDs starting at 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_anchor_monotonic_ids() {
        let mgr = make_manager();

        let (h1, _s1) = mgr.create_anchor::<u8>();
        let (h2, _s2) = mgr.create_anchor::<u8>();
        let (h3, _s3) = mgr.create_anchor::<u8>();

        let (_, id1) = h1.unpack();
        let (_, id2) = h2.unpack();
        let (_, id3) = h3.unpack();

        assert_eq!(id1, 1, "first local_id must be 1");
        assert_eq!(id2, 2, "second local_id must be 2");
        assert_eq!(id3, 3, "third local_id must be 3");
    }

    // -----------------------------------------------------------------------
    // Test 2: Registry contains entry after create_anchor
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_anchor_registry_insert() {
        let mgr = make_manager();

        let (handle, _stream) = mgr.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        assert!(
            mgr.registry.contains_key(&local_id),
            "entry must be present in registry after create_anchor"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Exclusive attach -- second attach while attached returns AlreadyAttached
    // -----------------------------------------------------------------------

    #[test]
    fn test_exclusive_attach() {
        let mgr = make_manager();
        let (handle, _stream) = mgr.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // First attach succeeds.
        let result1 = mgr.try_attach(local_id, flume::bounded::<Vec<u8>>(256).1, handle);
        assert!(result1.is_ok(), "first attach must succeed: {result1:?}");

        // Second attach while still attached must fail with AlreadyAttached.
        let result2 = mgr.try_attach(local_id, flume::bounded::<Vec<u8>>(256).1, handle);
        match result2 {
            Err(AttachError::AlreadyAttached { .. }) => {}
            other => panic!("expected AlreadyAttached, got {other:?}"),
        }

        // Detach and try again -- must succeed.
        let detached = mgr.detach(local_id);
        assert!(detached.is_some(), "detach must return the reader");

        let result3 = mgr.try_attach(local_id, flume::bounded::<Vec<u8>>(256).1, handle);
        assert!(result3.is_ok(), "third attach after detach must succeed: {result3:?}");
    }

    // -----------------------------------------------------------------------
    // Test 4: CancellationToken is idempotent across multiple cancel() calls
    // -----------------------------------------------------------------------

    #[test]
    fn test_cancel_token_idempotent() {
        let mgr = make_manager();
        let (handle, _stream) = mgr.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        // Retrieve a clone of the token before removing the entry.
        let token = mgr
            .registry
            .get(&local_id)
            .map(|e| e.cancel_token.clone())
            .expect("entry must exist");

        // First cancel -- should not panic.
        token.cancel();
        assert!(token.is_cancelled(), "token must be cancelled after first cancel()");

        // Second cancel -- must not panic and must still report cancelled.
        token.cancel();
        assert!(token.is_cancelled(), "token must still be cancelled after second cancel()");
    }

    // -----------------------------------------------------------------------
    // Test 5: remove_anchor removes the entry from the registry
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_cleanup() {
        let mgr = make_manager();
        let (handle, _stream) = mgr.create_anchor::<u8>();
        let (_, local_id) = handle.unpack();

        assert!(
            mgr.registry.contains_key(&local_id),
            "entry must exist before cleanup"
        );

        let removed = mgr.remove_anchor(local_id);
        assert!(removed.is_some(), "remove_anchor must return the entry");
        assert!(
            !mgr.registry.contains_key(&local_id),
            "entry must be absent after remove_anchor"
        );
    }

    // -----------------------------------------------------------------------
    // attach_stream_anchor tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_attach_stream_anchor_success() {
        let (mgr, _transport) = make_configurable_manager();
        let (handle, _stream) = mgr.create_anchor::<u32>();

        let result = mgr
            .attach_stream_anchor::<u32>(handle, "mock://endpoint", 1)
            .await;

        assert!(result.is_ok(), "attach_stream_anchor should succeed: {:?}", result.err());

        // The returned StreamSender should be usable
        let sender = result.unwrap();
        sender.finalize().expect("finalize should succeed");
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_not_found() {
        let (mgr, _transport) = make_configurable_manager();
        // Create a handle for a non-existent anchor
        let fake_handle = crate::handle::StreamAnchorHandle::pack(
            velo_common::WorkerId::from_u64(42),
            999,
        );

        let result = mgr
            .attach_stream_anchor::<u32>(fake_handle, "mock://endpoint", 1)
            .await;

        match result {
            Err(AttachError::AnchorNotFound { .. }) => {}
            other => panic!("expected AnchorNotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_already_attached() {
        let (mgr, _transport) = make_configurable_manager();
        let (handle, _stream) = mgr.create_anchor::<u32>();

        // First attach should succeed
        let sender1 = mgr
            .attach_stream_anchor::<u32>(handle, "mock://endpoint", 1)
            .await
            .expect("first attach should succeed");

        // Second attach should fail with AlreadyAttached
        let result = mgr
            .attach_stream_anchor::<u32>(handle, "mock://endpoint", 2)
            .await;

        match result {
            Err(AttachError::AlreadyAttached { .. }) => {}
            other => panic!("expected AlreadyAttached, got {:?}", other),
        }

        drop(sender1);
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_sender_can_send() {
        let (mgr, _transport) = make_configurable_manager();
        let (handle, stream) = mgr.create_anchor::<u32>();

        let sender = mgr
            .attach_stream_anchor::<u32>(handle, "mock://endpoint", 1)
            .await
            .expect("attach should succeed");

        // Send an item through the StreamSender
        sender.send(42u32).await.expect("send should succeed");

        // The item should arrive on the AnchorStream's internal receiver
        let bytes = stream.rx.try_recv().expect("should receive item on stream rx");
        let frame: crate::frame::StreamFrame<u32> =
            rmp_serde::from_slice(&bytes).expect("deserialize");
        match frame {
            crate::frame::StreamFrame::Item(val) => assert_eq!(val, 42),
            other => panic!("expected Item(42), got {:?}", other),
        }

        drop(sender);
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_transport_error() {
        let (mgr, transport) = make_configurable_manager();
        let (handle, _stream) = mgr.create_anchor::<u32>();

        // Configure transport to fail on connect
        transport.set_connect_fail(true);

        let result = mgr
            .attach_stream_anchor::<u32>(handle, "mock://endpoint", 1)
            .await;

        match result {
            Err(AttachError::TransportError(_)) => {}
            other => panic!("expected TransportError, got {:?}", other),
        }
    }
}
