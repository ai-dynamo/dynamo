// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anchor registry layer: [`AnchorManager`], `AnchorEntry`, [`StreamAnchor`], and [`AttachError`].
//!
//! The anchor registry is the core coordination point for the streaming protocol.
//! Each anchor represents a single exclusive-attachment stream slot:
//!
//! - [`AnchorManager::create_anchor`] allocates a registry slot and returns a
//!   [`StreamAnchor<T>`] that embeds the [`crate::handle::StreamAnchorHandle`]
//!   (obtainable via [`.handle()`](StreamAnchor::handle)) for the consumer.
//! - Exactly one [`flume::Sender`] may be attached at a time;
//!   the attach check is performed atomically via [`dashmap::DashMap::entry`].
//! - Each entry holds a [`tokio_util::sync::CancellationToken`] created at anchor
//!   creation so that whichever cleanup path fires first cancels the token; subsequent
//!   cancellations are no-ops.

use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::task::{Context, Poll};
use std::time::Duration;

use dashmap::DashMap;
use derive_builder::Builder;
use futures::Stream;
use serde::de::DeserializeOwned;
use tokio_util::sync::CancellationToken;

use crate::frame::{StreamError, StreamFrame};
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
/// The `attachment` flag indicates whether a sender is currently attached.
/// The check-and-set is performed atomically via [`dashmap::mapref::entry::Entry`]
/// to prevent TOCTOU races. The reader pump takes ownership of the transport
/// receiver directly rather than storing it in the entry.
// Fields are consumed by Phase 7+ control handlers and Phase 8 data path.
#[allow(dead_code)]
pub(crate) struct AnchorEntry {
    /// Raw-bytes frame delivery channel to the [`StreamAnchor<T>`] consumer.
    ///
    /// Non-generic so `DashMap<u64, AnchorEntry>` requires no type parameters.
    pub frame_tx: flume::Sender<Vec<u8>>,

    /// Created at anchor creation; cancelled by whichever cleanup path fires first.
    ///
    /// `cancel()` is idempotent -- a second caller is a no-op.
    pub cancel_token: CancellationToken,

    /// `true` iff a sender is currently attached. The reader pump owns the
    /// transport receiver separately (not stored here).
    pub attachment: bool,

    /// Cancels the inactivity timeout task when a sender attaches.
    /// `None` if no timeout is configured for this anchor.
    pub timeout_cancel: Option<CancellationToken>,

    /// The configured timeout duration for this anchor. Stored so that
    /// `detach` can respawn the timeout task with the same duration.
    pub timeout_duration: Option<Duration>,

    /// Populated on successful attach from [`crate::control::AnchorAttachRequest::stream_cancel_handle`].
    /// Encodes the sender's WorkerId + stream ID so the anchor can route `_stream_cancel`
    /// active messages to the correct sender worker when the consumer cancels upstream.
    /// `None` until a sender attaches.
    pub stream_cancel_handle: Option<crate::control::StreamCancelHandle>,
}

// ---------------------------------------------------------------------------
// StreamController
// ---------------------------------------------------------------------------

/// Shared inner state between [`StreamAnchor`] and [`StreamController`].
///
/// Wrapped in `Arc` so `StreamController` can outlive `StreamAnchor` being
/// moved into StreamExt combinators.
struct StreamControllerInner {
    local_id: u64,
    registry: Arc<DashMap<u64, AnchorEntry>>,
    /// Sender-side registry: used to directly cancel the [`crate::control::SenderEntry`]
    /// when the anchor is cancelled (same-worker path without AM round-trip).
    sender_registry: Arc<crate::control::SenderRegistry>,
    /// Optional messenger for sending `_stream_cancel` AM to the sender's worker.
    /// `None` for local-only (MockFrameTransport) scenarios.
    messenger: Option<Arc<velo_messenger::Messenger>>,
    /// AtomicBool gate: compare_exchange(false, true) to ensure AM is sent at most once.
    cancelled: AtomicBool,
}

/// Cloneable handle to cancel a [`StreamAnchor`] from outside the stream.
///
/// Obtain via [`StreamAnchor::controller`]. Required for the StreamExt combinator
/// use-case where the `StreamAnchor` is moved into `.map()` / `.take_while()` etc.
/// and the caller loses direct access to it.
#[derive(Clone)]
pub struct StreamController {
    inner: Arc<StreamControllerInner>,
}

impl StreamController {
    /// Cancel the stream: remove the anchor from the registry and send a
    /// `_stream_cancel` AM to the sender's worker (fire-and-forget).
    ///
    /// Idempotent: the AM is sent at most once regardless of how many clones
    /// call `cancel()` concurrently.
    pub fn cancel(&self) {
        // AtomicBool gate: only the first caller proceeds
        if self
            .inner
            .cancelled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return; // already cancelled
        }

        // Remove anchor from registry and extract stream_cancel_handle
        let stream_cancel_handle = self
            .inner
            .registry
            .remove(&self.inner.local_id)
            .and_then(|(_, entry)| {
                entry.cancel_token.cancel();
                entry.stream_cancel_handle
            });

        // Directly cancel the SenderEntry in the local sender_registry.
        // This fires the user-facing cancel_token and poisons send() immediately
        // without requiring an AM round-trip. Idempotent: remove returns None if
        // the entry was already removed (e.g. finalize/detach ran first).
        if let Some(handle) = stream_cancel_handle {
            let (sender_worker_id, sender_stream_id) = handle.unpack();
            if let Some((_, entry)) = self.inner.sender_registry.senders.remove(&sender_stream_id) {
                drop(entry.rx_closer.lock().unwrap().take());
                entry.cancel_token.cancel();
            }

            // Also send _stream_cancel AM for cross-worker scenarios (messenger present)
            if let Some(messenger) = self.inner.messenger.clone() {
                let payload =
                    serde_json::to_vec(&crate::control::StreamCancelRequest { sender_stream_id })
                        .expect("serialize StreamCancelRequest");
                // Fire-and-forget: use tokio::spawn guarded by try_current()
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.spawn(async move {
                        let _ = messenger
                            .am_send_streaming("_stream_cancel")
                            .expect("am_send_streaming builder")
                            .raw_payload(bytes::Bytes::from(payload))
                            .worker(sender_worker_id)
                            .send()
                            .await;
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamAnchor<T>
// ---------------------------------------------------------------------------

/// Consumer-side receive stream for an anchor.
///
/// Implements [`futures::Stream`] yielding `Result<StreamFrame<T>, StreamError>`.
/// Heartbeat frames are filtered out and never exposed to the consumer.
/// Terminal sentinels (`Finalized`, `Detached`, `Dropped`, `TransportError`)
/// cause the stream to yield one final item and then `None` on subsequent polls.
///
/// Use [`StreamExt::next()`](futures::StreamExt::next) for async iteration.
///
/// # Example
///
/// ```rust,no_run
/// use futures::StreamExt;
/// use velo_streaming::{AnchorManager, StreamFrame};
///
/// # async fn example(mgr: &AnchorManager) -> anyhow::Result<()> {
/// // Consumer creates an anchor
/// let mut anchor = mgr.create_anchor::<String>();
/// let handle = anchor.handle();
///
/// // Producer attaches (could be on a different worker)
/// let sender = mgr.attach_stream_anchor::<String>(handle).await?;
///
/// // Send items
/// sender.send("hello".into()).await?;
/// sender.send("world".into()).await?;
/// sender.finalize()?;
///
/// // Consume the stream
/// while let Some(frame) = anchor.next().await {
///     match frame {
///         Ok(StreamFrame::Item(s)) => println!("{s}"),
///         Ok(StreamFrame::Finalized) => break,
///         Err(e) => eprintln!("stream error: {e}"),
///         _ => {}
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// For upstream cancellation, see [`StreamController`].
pub struct StreamAnchor<T> {
    /// The anchor handle — pass to a sender for attachment via
    /// [`AnchorManager::attach_stream_anchor`].
    handle: StreamAnchorHandle,
    /// Async stream obtained from consuming the flume::Receiver via `into_stream()`.
    inner_stream: flume::r#async::RecvStream<'static, Vec<u8>>,
    /// Set to true after a terminal sentinel; prevents further polling.
    terminated: bool,
    /// The local ID of the anchor in the registry (for cancel).
    local_id: u64,
    /// Arc clone of the AnchorManager's registry (for cancel).
    registry: Arc<DashMap<u64, AnchorEntry>>,
    /// Shared cancel handle — also held by any [`StreamController`] clones.
    controller: StreamController,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> StreamAnchor<T> {
    pub(crate) fn new(
        handle: StreamAnchorHandle,
        rx: flume::Receiver<Vec<u8>>,
        local_id: u64,
        registry: Arc<DashMap<u64, AnchorEntry>>,
        sender_registry: Arc<crate::control::SenderRegistry>,
        messenger: Option<Arc<velo_messenger::Messenger>>,
    ) -> Self {
        let inner = Arc::new(StreamControllerInner {
            local_id,
            registry: registry.clone(),
            sender_registry,
            messenger,
            cancelled: AtomicBool::new(false),
        });
        let controller = StreamController { inner };
        Self {
            handle,
            inner_stream: rx.into_stream(),
            terminated: false,
            local_id,
            registry,
            controller,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return the anchor handle. Pass to a sender (possibly on another worker)
    /// for attachment via [`AnchorManager::attach_stream_anchor`].
    pub fn handle(&self) -> StreamAnchorHandle {
        self.handle
    }

    /// Return a cloneable [`StreamController`] that can cancel this anchor
    /// even after `self` is moved into a StreamExt combinator.
    pub fn controller(&self) -> StreamController {
        self.controller.clone()
    }

    /// Consume the stream and cancel the anchor.
    ///
    /// Removes the anchor from the registry and sends `_stream_cancel` AM to
    /// the sender's worker if a sender is attached. Same effect as
    /// [`StreamController::cancel`] but consumes `self` to signal intent.
    pub fn cancel(mut self) -> StreamController {
        self.terminated = true; // prevent Drop from re-cancelling
        self.controller.cancel();
        self.controller.clone()
    }

    /// Configure or override the inactivity timeout for this anchor.
    ///
    /// - `Some(duration)`: anchor will be auto-removed if no sender attaches
    ///   within `duration`. If the anchor is currently unattached, a new timeout
    ///   task is spawned immediately (replacing any existing one).
    /// - `None`: disable timeout for this anchor. Any running timeout task is
    ///   cancelled.
    ///
    /// If the anchor is currently attached, the new duration is stored and will
    /// take effect on the next detach (no immediate spawn since the timer is
    /// paused while attached).
    pub fn set_timeout(&self, timeout: Option<Duration>) {
        if let Some(mut entry) = self.registry.get_mut(&self.local_id) {
            // Cancel existing timeout task if any
            if let Some(ref old_tc) = entry.timeout_cancel {
                old_tc.cancel();
            }

            // Update the stored duration
            entry.timeout_duration = timeout;

            // If unattached and a timeout is set, spawn a new timeout task
            if !entry.attachment {
                if let Some(duration) = timeout {
                    let tc = AnchorManager::spawn_timeout_task(
                        self.registry.clone(),
                        self.local_id,
                        duration,
                    );
                    entry.timeout_cancel = Some(tc);
                } else {
                    entry.timeout_cancel = None;
                }
            } else {
                // Attached: just clear the old cancel token; duration is stored
                // and will be used when detach respawns the timeout task.
                entry.timeout_cancel = None;
            }
        }
    }
}

// SAFETY: StreamAnchor does not use structural pinning. Its `inner_stream`
// (flume::r#async::RecvStream) is Unpin, and all other fields are trivially Unpin.
// PhantomData<T> should not prevent Unpin, but we assert it explicitly.
impl<T> Unpin for StreamAnchor<T> {}

impl<T> Drop for StreamAnchor<T> {
    fn drop(&mut self) {
        if !self.terminated {
            // Delegate to the shared controller — AtomicBool prevents double-cancel.
            self.controller.cancel();
        }
    }
}

impl<T: DeserializeOwned> Stream for StreamAnchor<T> {
    type Item = Result<StreamFrame<T>, StreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.terminated {
            return Poll::Ready(None);
        }
        loop {
            match Pin::new(&mut this.inner_stream).poll_next(cx) {
                Poll::Ready(Some(bytes)) => {
                    match rmp_serde::from_slice::<StreamFrame<T>>(&bytes) {
                        Ok(StreamFrame::Heartbeat) => continue, // filter heartbeats
                        Ok(StreamFrame::Item(data)) => {
                            return Poll::Ready(Some(Ok(StreamFrame::Item(data))));
                        }
                        Ok(StreamFrame::SenderError(msg)) => {
                            // Soft error -- stream continues (not terminated)
                            return Poll::Ready(Some(Err(StreamError::SenderError(msg))));
                        }
                        Ok(StreamFrame::Finalized) => {
                            this.terminated = true;
                            // Clean up registry entry — anchor is permanently closed.
                            if let Some((_, entry)) = this.registry.remove(&this.local_id) {
                                entry.cancel_token.cancel();
                            }
                            return Poll::Ready(Some(Ok(StreamFrame::Finalized)));
                        }
                        Ok(StreamFrame::Detached) => {
                            // Detached is NOT terminal — a new sender may reattach.
                            // Clear the attachment flag so attach_stream_anchor can succeed.
                            if let Some(mut entry) = this.registry.get_mut(&this.local_id) {
                                entry.attachment = false;
                            }
                            return Poll::Ready(Some(Ok(StreamFrame::Detached)));
                        }
                        Ok(StreamFrame::Dropped) => {
                            this.terminated = true;
                            // Clean up registry entry — sender dropped without explicit close.
                            if let Some((_, entry)) = this.registry.remove(&this.local_id) {
                                entry.cancel_token.cancel();
                            }
                            return Poll::Ready(Some(Err(StreamError::SenderDropped)));
                        }
                        Ok(StreamFrame::TransportError(msg)) => {
                            this.terminated = true;
                            return Poll::Ready(Some(Err(StreamError::TransportError(msg))));
                        }
                        Err(e) => {
                            this.terminated = true;
                            return Poll::Ready(Some(Err(
                                StreamError::DeserializationError(e.to_string()),
                            )));
                        }
                    }
                }
                Poll::Ready(None) => {
                    this.terminated = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
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
///
/// Use [`AnchorManagerBuilder`] for optional configuration (e.g. `default_timeout`),
/// or [`AnchorManager::new`] as a convenience constructor with no timeout.
#[derive(Builder)]
#[builder(pattern = "owned", build_fn(name = "build_inner", private))]
pub struct AnchorManager {
    worker_id: velo_common::WorkerId,

    #[builder(setter(skip), default = "AtomicU64::new(0)")]
    next_local_id: AtomicU64,

    #[builder(default = "Arc::new(DashMap::new())")]
    pub(crate) registry: Arc<DashMap<u64, AnchorEntry>>,

    pub transport: Arc<dyn crate::transport::FrameTransport>,

    /// Optional inactivity timeout for newly created anchors.
    /// When set, `create_anchor` spawns a timeout task that auto-removes the
    /// anchor if no sender attaches within this duration.
    #[builder(default, setter(into, strip_option))]
    pub default_timeout: Option<Duration>,

    /// Optional messenger for sending `_stream_cancel` AM from the consumer side.
    /// Set by VeloFrameTransport scenarios; `None` for local/mock transport scenarios.
    #[builder(default)]
    pub messenger: Option<Arc<velo_messenger::Messenger>>,

    /// Monotonically increasing counter for sender_stream_id values.
    /// Separate from next_local_id to keep anchor-side and sender-side namespaces distinct.
    #[builder(setter(skip), default = "AtomicU64::new(0)")]
    next_sender_stream_id: AtomicU64,

    /// Sender-side registry: maps sender_stream_id -> SenderEntry.
    /// Shared with the _stream_cancel handler registered on this AnchorManager.
    /// Also accessed by StreamSender::Drop / finalize / detach for cleanup.
    #[builder(default = "Arc::new(crate::control::SenderRegistry::default())")]
    pub sender_registry: Arc<crate::control::SenderRegistry>,

    /// Write-once lock storing the live Messenger after `register_handlers` is called.
    /// `None` until `register_handlers` succeeds; subsequent calls return `Err`.
    #[builder(setter(skip), default = "std::sync::OnceLock::new()")]
    pub(crate) messenger_lock: std::sync::OnceLock<Arc<velo_messenger::Messenger>>,
}

impl AnchorManagerBuilder {
    /// Build the [`AnchorManager`].
    pub fn build(self) -> Result<AnchorManager, AnchorManagerBuilderError> {
        self.build_inner()
    }
}

impl AnchorManager {
    /// Convenience constructor with no default timeout.
    ///
    /// Equivalent to `AnchorManagerBuilder::default().worker_id(id).transport(t).build()`.
    pub fn new(
        worker_id: velo_common::WorkerId,
        transport: Arc<dyn crate::transport::FrameTransport>,
    ) -> Self {
        AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .build()
            .expect("required fields provided")
    }

    /// Allocate a new anchor and return a [`StreamAnchor<T>`] for the consumer.
    ///
    /// The returned `StreamAnchor` embeds the [`StreamAnchorHandle`]; obtain it via
    /// [`.handle()`](StreamAnchor::handle) to pass to a sender for attachment.
    ///
    /// Local IDs start at 1 and increment monotonically; ID 0 is reserved.
    /// A flume bounded channel (capacity 256) is created per anchor to deliver raw frame bytes.
    ///
    /// If `default_timeout` is configured, a background task is spawned that will
    /// auto-remove the anchor if no sender attaches within the timeout duration.
    pub fn create_anchor<T>(&self) -> StreamAnchor<T> {
        // fetch_add returns the *old* value (starts at 0), so +1 gives us IDs starting at 1.
        let local_id = self.next_local_id.fetch_add(1, Ordering::Relaxed) + 1;

        let (frame_tx, frame_rx) = flume::bounded::<Vec<u8>>(256);
        let cancel_token = CancellationToken::new();

        // Spawn timeout task if default_timeout is configured
        let timeout_cancel = self.default_timeout.map(|timeout| {
            Self::spawn_timeout_task(self.registry.clone(), local_id, timeout)
        });

        let entry = AnchorEntry {
            frame_tx,
            cancel_token,
            attachment: false,
            timeout_cancel,
            timeout_duration: self.default_timeout,
            stream_cancel_handle: None, // populated on attach
        };

        self.registry.insert(local_id, entry);

        let handle = StreamAnchorHandle::pack(self.worker_id, local_id);
        StreamAnchor::new(
            handle,
            frame_rx,
            local_id,
            self.registry.clone(),
            self.sender_registry.clone(),
            self.messenger.clone(),
        )
    }

    /// Spawn a background task that removes the anchor after `timeout` elapses.
    ///
    /// Returns a [`CancellationToken`] that cancels the task when triggered
    /// (e.g. on attach, or when `set_timeout(None)` is called).
    pub(crate) fn spawn_timeout_task(
        registry: Arc<DashMap<u64, AnchorEntry>>,
        local_id: u64,
        timeout: Duration,
    ) -> CancellationToken {
        let tc = CancellationToken::new();
        let tc_clone = tc.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = tc_clone.cancelled() => {
                    // Attach or explicit cancel -- do nothing
                }
                _ = tokio::time::sleep(timeout) => {
                    // Timeout expired -- remove anchor
                    if let Some((_, entry)) = registry.remove(&local_id) {
                        entry.cancel_token.cancel();
                        // Dropping frame_tx closes the channel -> StreamAnchor yields None
                    }
                }
            }
        });
        tc
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

    /// Atomically attempt to mark an anchor as attached.
    ///
    /// Uses `DashMap::entry()` to perform the check-and-set atomically under
    /// the shard lock, preventing TOCTOU races between concurrent attach attempts.
    /// The reader pump takes ownership of the transport receiver separately.
    ///
    /// If a timeout task is running, it is cancelled (paused) on successful attach.
    ///
    /// Returns `Err(AttachError::AlreadyAttached)` if a sender is already attached.
    /// Returns `Err(AttachError::AnchorNotFound)` if `local_id` is not in the registry.
    #[allow(dead_code)]
    pub(crate) fn try_attach(
        &self,
        local_id: u64,
        handle: StreamAnchorHandle,
    ) -> Result<(), AttachError> {
        use dashmap::mapref::entry::Entry;
        match self.registry.entry(local_id) {
            Entry::Vacant(_) => Err(AttachError::AnchorNotFound { handle }),
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment {
                    Err(AttachError::AlreadyAttached { handle })
                } else {
                    entry.attachment = true;
                    // Cancel the timeout task while attached (pause timer)
                    if let Some(ref tc) = entry.timeout_cancel {
                        tc.cancel();
                    }
                    Ok(())
                }
            }
        }
    }

    /// Clear the attachment flag on an anchor.
    ///
    /// If the anchor has a configured `timeout_duration`, a new timeout task
    /// is spawned (timer "resumes" by restarting from the full duration).
    ///
    /// Returns `true` if the anchor was found and was previously attached.
    #[allow(dead_code)]
    pub(crate) fn detach(&self, local_id: u64) -> bool {
        // Phase 1: Clear attachment and read timeout_duration (drop DashMap ref)
        let (was_attached, maybe_timeout) = self
            .registry
            .get_mut(&local_id)
            .map(|mut entry| {
                let was = entry.attachment;
                entry.attachment = false;
                (was, entry.timeout_duration)
            })
            .unwrap_or((false, None));

        // Phase 2: Respawn timeout task outside the DashMap borrow
        if let Some(timeout) = maybe_timeout {
            let tc = Self::spawn_timeout_task(self.registry.clone(), local_id, timeout);
            // Store the new cancellation token back in the entry
            if let Some(mut entry) = self.registry.get_mut(&local_id) {
                entry.timeout_cancel = Some(tc);
            }
        }

        was_attached
    }

    /// Register all five control-plane AM handlers on a live Messenger.
    ///
    /// Registers: `_anchor_attach`, `_anchor_detach`, `_anchor_finalize`,
    /// `_anchor_cancel` (all on `self` as `Arc<AnchorManager>`), and
    /// `_stream_cancel` (on `self.sender_registry`).
    ///
    /// Stores the messenger in `messenger_lock` (write-once) for use by
    /// `attach_remote` in Phase 12 Plan 02.
    ///
    /// # Errors
    ///
    /// Returns `Err` if called twice (OnceLock already set) or if any
    /// handler registration fails (e.g., duplicate handler name).
    ///
    /// # Panics
    ///
    /// Does not panic. Caller must hold an `Arc<AnchorManager>`.
    pub fn register_handlers(
        self: &Arc<Self>,
        messenger: Arc<velo_messenger::Messenger>,
    ) -> anyhow::Result<()> {
        use crate::control::{
            create_anchor_attach_handler,
            create_anchor_cancel_handler,
            create_anchor_detach_handler,
            create_anchor_finalize_handler,
            create_stream_cancel_handler,
        };

        messenger.register_streaming_handler(
            create_anchor_attach_handler(Arc::clone(self))
        )?;
        messenger.register_streaming_handler(
            create_anchor_detach_handler(Arc::clone(self))
        )?;
        messenger.register_streaming_handler(
            create_anchor_finalize_handler(Arc::clone(self))
        )?;
        messenger.register_streaming_handler(
            create_anchor_cancel_handler(Arc::clone(self))
        )?;
        messenger.register_streaming_handler(
            create_stream_cancel_handler(Arc::clone(&self.sender_registry))
        )?;

        self.messenger_lock
            .set(messenger)
            .map_err(|_| anyhow::anyhow!("register_handlers called twice"))?;

        Ok(())
    }

    /// Attach a sender to an existing anchor via the remote control-plane path.
    ///
    /// Called when `attach_stream_anchor` detects that `handle.worker_id != self.worker_id`.
    ///
    /// Sends an `_anchor_attach` AM to the remote worker, receives the stream endpoint,
    /// calls `transport.connect()` to establish the write channel, and returns a
    /// [`StreamSender<T>`](crate::sender::StreamSender) that writes directly into the
    /// transport bridge (which the remote reader_pump forwards to the anchor's frame channel).
    ///
    /// # Errors
    /// - [`AttachError::TransportError`] if `messenger_lock` is not set (register_handlers not called)
    /// - [`AttachError::TransportError`] if the AM send or transport connect fails
    /// - [`AttachError::TransportError`] if the remote worker returns `AnchorAttachResponse::Err`
    async fn attach_remote<T: serde::Serialize>(
        &self,
        handle: StreamAnchorHandle,
    ) -> Result<crate::sender::StreamSender<T>, AttachError> {
        let (handle_worker_id, _) = handle.unpack();

        // Require messenger_lock to be set (register_handlers must have been called)
        let messenger = self.messenger_lock.get().ok_or_else(|| {
            AttachError::TransportError(anyhow::anyhow!(
                "register_handlers not called — messenger unavailable for remote attach"
            ))
        })?;

        // Allocate sender_stream_id and build cancel infrastructure (same as local path)
        let sender_stream_id = self.next_sender_stream_id.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let (poison_tx, poison_rx) = flume::bounded::<()>(1);

        let stream_cancel_handle =
            crate::control::StreamCancelHandle::pack(self.worker_id, sender_stream_id);

        // Build request payload (serde_json — typed_unary_async handlers use JSON)
        // Use sender_stream_id as the session_id for the remote attach request.
        let req = crate::control::AnchorAttachRequest {
            handle,
            session_id: sender_stream_id,
            stream_cancel_handle,
        };

        // Send _anchor_attach AM to the remote worker (typed request-response)
        let response: crate::control::AnchorAttachResponse = messenger
            .typed_unary_streaming::<crate::control::AnchorAttachResponse>("_anchor_attach")
            .payload(&req)
            .map_err(AttachError::TransportError)?
            .worker(handle_worker_id)
            .send()
            .await
            .map_err(AttachError::TransportError)?;

        match response {
            crate::control::AnchorAttachResponse::Ok { stream_endpoint } => {
                let (_, local_id) = handle.unpack();

                // Connect transport to Worker A's endpoint — get frame_tx pump channel
                let frame_tx = self
                    .transport
                    .connect(&stream_endpoint, local_id, sender_stream_id)
                    .await?; // maps to AttachError::TransportError via From<anyhow::Error>

                // Register SenderEntry for _stream_cancel routing
                let sender_entry = crate::control::SenderEntry {
                    cancel_token: cancel_token.clone(),
                    rx_closer: std::sync::Mutex::new(Some(poison_rx)),
                };
                self.sender_registry.senders.insert(sender_stream_id, sender_entry);

                // Build StreamSender: frame_tx from transport.connect() (not local registry frame_tx)
                // No local AnchorEntry is created for Worker A's anchor on Worker B.
                Ok(crate::sender::StreamSender::new(
                    frame_tx,
                    handle,
                    self.registry.clone(), // Worker B's registry (no entry for this handle — correct)
                    cancel_token,
                    sender_stream_id,
                    self.sender_registry.clone(),
                    poison_tx,
                ))
            }
            crate::control::AnchorAttachResponse::Err { reason } => {
                Err(AttachError::TransportError(anyhow::anyhow!("{}", reason)))
            }
        }
    }

    /// Attach a sender to an existing anchor, establishing the transport connection.
    ///
    /// This is the primary sender-side entry point (API-05). It:
    /// 1. Detects remote handles (`handle.worker_id != self.worker_id`) and routes through
    ///    the remote attach path for cross-worker AM dispatch.
    /// 2. For local handles: validates the anchor exists and is unattached,
    ///    atomically marks the anchor as attached, and returns a
    ///    [`StreamSender<T>`](crate::sender::StreamSender) for pushing typed frames.
    ///
    /// The StreamSender writes to the entry's `frame_tx` so items flow directly
    /// to the [`StreamAnchor<T>`] consumer. The transport connection is used by the
    /// reader pump for cross-worker flows.
    ///
    /// # Errors
    /// - [`AttachError::AnchorNotFound`] if the handle is not in the registry (local path)
    /// - [`AttachError::AlreadyAttached`] if another sender is already connected (local path)
    /// - [`AttachError::TransportError`] for all remote path errors (messenger unavailable,
    ///   AM send failed, remote error response)
    pub async fn attach_stream_anchor<T: serde::Serialize>(
        &self,
        handle: StreamAnchorHandle,
    ) -> Result<crate::sender::StreamSender<T>, AttachError> {
        let (handle_worker_id, local_id) = handle.unpack();

        // Remote path: handle belongs to a different worker — send _anchor_attach AM
        if handle_worker_id != self.worker_id {
            return self.attach_remote::<T>(handle).await;
        }

        // Step 1: Quick check anchor exists and is unattached (drop ref before async)
        {
            let entry = self.registry.get(&local_id);
            match entry {
                None => return Err(AttachError::AnchorNotFound { handle }),
                Some(e) if e.attachment => {
                    return Err(AttachError::AlreadyAttached { handle });
                }
                _ => {} // looks good, proceed
            }
        } // DashMap ref dropped here

        // Step 2: Atomically set attachment under shard lock.
        // Re-check under the entry guard to prevent TOCTOU.
        use dashmap::mapref::entry::Entry;
        match self.registry.entry(local_id) {
            Entry::Vacant(_) => Err(AttachError::AnchorNotFound { handle }),
            Entry::Occupied(mut occ) => {
                let entry = occ.get_mut();
                if entry.attachment {
                    Err(AttachError::AlreadyAttached { handle })
                } else {
                    // Clone the frame_tx so the StreamSender can write items
                    // directly to the StreamAnchor consumer.
                    let frame_tx = entry.frame_tx.clone();

                    // Mark as attached (reader pump takes ownership of transport
                    // receiver separately).
                    entry.attachment = true;

                    // Cancel the timeout task while attached (pause timer)
                    if let Some(ref tc) = entry.timeout_cancel {
                        tc.cancel();
                    }

                    // Allocate sender_stream_id and build SenderEntry
                    let sender_stream_id = self.next_sender_stream_id.fetch_add(1, Ordering::Relaxed) + 1;
                    let cancel_token = tokio_util::sync::CancellationToken::new();
                    let (poison_tx, poison_rx) = flume::bounded::<()>(1);

                    let sender_entry = crate::control::SenderEntry {
                        cancel_token: cancel_token.clone(),
                        rx_closer: std::sync::Mutex::new(Some(poison_rx)),
                    };
                    self.sender_registry.senders.insert(sender_stream_id, sender_entry);

                    // Store stream_cancel_handle in AnchorEntry (already under DashMap lock)
                    entry.stream_cancel_handle = Some(crate::control::StreamCancelHandle::pack(
                        self.worker_id, sender_stream_id
                    ));

                    // Return sender with all new fields
                    Ok(crate::sender::StreamSender::new(
                        frame_tx,
                        handle,
                        self.registry.clone(),
                        cancel_token,
                        sender_stream_id,
                        self.sender_registry.clone(),
                        poison_tx,
                    ))
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
    use futures::StreamExt;
    use std::sync::Arc;
    use crate::frame::{StreamFrame, StreamError};

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

    // -----------------------------------------------------------------------
    // Test 1: Monotonic local IDs starting at 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_anchor_monotonic_ids() {
        let mgr = make_manager();

        let a1 = mgr.create_anchor::<u8>();
        let a2 = mgr.create_anchor::<u8>();
        let a3 = mgr.create_anchor::<u8>();

        let (_, id1) = a1.handle().unpack();
        let (_, id2) = a2.handle().unpack();
        let (_, id3) = a3.handle().unpack();

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

        let anchor = mgr.create_anchor::<u8>();
        let (_, local_id) = anchor.handle().unpack();

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
        let anchor = mgr.create_anchor::<u8>();
        let handle = anchor.handle();
        let (_, local_id) = handle.unpack();

        // First attach succeeds.
        let result1 = mgr.try_attach(local_id, handle);
        assert!(result1.is_ok(), "first attach must succeed: {result1:?}");

        // Second attach while still attached must fail with AlreadyAttached.
        let result2 = mgr.try_attach(local_id, handle);
        match result2 {
            Err(AttachError::AlreadyAttached { .. }) => {}
            other => panic!("expected AlreadyAttached, got {other:?}"),
        }

        // Detach and try again -- must succeed.
        let was_attached = mgr.detach(local_id);
        assert!(was_attached, "detach must return true when attached");

        let result3 = mgr.try_attach(local_id, handle);
        assert!(result3.is_ok(), "third attach after detach must succeed: {result3:?}");
    }

    // -----------------------------------------------------------------------
    // Test 4: CancellationToken is idempotent across multiple cancel() calls
    // -----------------------------------------------------------------------

    #[test]
    fn test_cancel_token_idempotent() {
        let mgr = make_manager();
        let anchor = mgr.create_anchor::<u8>();
        let (_, local_id) = anchor.handle().unpack();

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
        let anchor = mgr.create_anchor::<u8>();
        let (_, local_id) = anchor.handle().unpack();

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
        let mgr = make_manager();
        let anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();

        let result = mgr
            .attach_stream_anchor::<u32>(handle)
            .await;

        assert!(result.is_ok(), "attach_stream_anchor should succeed: {:?}", result.err());

        // The returned StreamSender should be usable
        let sender = result.unwrap();
        sender.finalize().expect("finalize should succeed");
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_not_found() {
        let mgr = make_manager();
        // Create a handle for a non-existent anchor
        let fake_handle = crate::handle::StreamAnchorHandle::pack(
            velo_common::WorkerId::from_u64(42),
            999,
        );

        let result = mgr
            .attach_stream_anchor::<u32>(fake_handle)
            .await;

        match result {
            Err(AttachError::AnchorNotFound { .. }) => {}
            other => panic!("expected AnchorNotFound, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_already_attached() {
        let mgr = make_manager();
        let anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();

        // First attach should succeed
        let sender1 = mgr
            .attach_stream_anchor::<u32>(handle)
            .await
            .expect("first attach should succeed");

        // Second attach should fail with AlreadyAttached
        let result = mgr
            .attach_stream_anchor::<u32>(handle)
            .await;

        match result {
            Err(AttachError::AlreadyAttached { .. }) => {}
            other => panic!("expected AlreadyAttached, got {:?}", other),
        }

        drop(sender1);
    }

    #[tokio::test]
    async fn test_attach_stream_anchor_sender_can_send() {
        let mgr = make_manager();
        let mut anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();

        let sender = mgr
            .attach_stream_anchor::<u32>(handle)
            .await
            .expect("attach should succeed");

        // Send an item through the StreamSender
        sender.send(42u32).await.expect("send should succeed");

        // The item should arrive via the Stream interface
        let result = anchor.next().await;
        match result {
            Some(Ok(StreamFrame::Item(val))) => assert_eq!(val, 42),
            other => panic!("expected Item(42), got {:?}", other),
        }

        drop(sender);
    }

    // -----------------------------------------------------------------------
    // StreamAnchor<T> Stream impl tests (Plan 08-03, Task 1)
    // -----------------------------------------------------------------------

    /// Helper: create a raw channel pair + StreamAnchor for testing Stream impl.
    /// Returns (sender for pushing raw bytes, StreamAnchor<T>).
    fn make_test_stream<T>() -> (flume::Sender<Vec<u8>>, StreamAnchor<T>) {
        let mgr = make_manager();
        let anchor = mgr.create_anchor::<T>();
        let (_, local_id) = anchor.handle().unpack();
        // Get the frame_tx from the registry for pushing raw bytes
        let frame_tx = mgr
            .registry
            .get(&local_id)
            .map(|e| e.frame_tx.clone())
            .expect("entry must exist");
        (frame_tx, anchor)
    }

    #[tokio::test]
    async fn test_stream_yields_item() {
        let (tx, mut stream) = make_test_stream::<u32>();

        // Send serialized Item frame
        let bytes = rmp_serde::to_vec(&StreamFrame::Item(42u32)).unwrap();
        tx.send(bytes).unwrap();

        let result = stream.next().await;
        match result {
            Some(Ok(StreamFrame::Item(val))) => assert_eq!(val, 42),
            other => panic!("expected Some(Ok(Item(42))), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_stream_yields_sender_error_and_continues() {
        let (tx, mut stream) = make_test_stream::<u32>();

        // Send SenderError
        let err_bytes = rmp_serde::to_vec(&StreamFrame::<u32>::SenderError("oops".to_string())).unwrap();
        tx.send(err_bytes).unwrap();

        // Should yield Err(StreamError::SenderError)
        let result = stream.next().await;
        match result {
            Some(Err(StreamError::SenderError(msg))) => assert_eq!(msg, "oops"),
            other => panic!("expected SenderError, got {:?}", other),
        }

        // Stream should continue -- send another item
        let item_bytes = rmp_serde::to_vec(&StreamFrame::Item(99u32)).unwrap();
        tx.send(item_bytes).unwrap();

        let result2 = stream.next().await;
        match result2 {
            Some(Ok(StreamFrame::Item(val))) => assert_eq!(val, 99),
            other => panic!("expected Item(99) after SenderError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_stream_finalized_then_none() {
        let (tx, mut stream) = make_test_stream::<u32>();

        let bytes = rmp_serde::to_vec(&StreamFrame::<u32>::Finalized).unwrap();
        tx.send(bytes).unwrap();

        // Should yield Ok(Finalized)
        let result = stream.next().await;
        assert!(matches!(result, Some(Ok(StreamFrame::Finalized))), "expected Finalized, got {:?}", result);

        // Next call should yield None
        let result2 = stream.next().await;
        assert!(result2.is_none(), "expected None after Finalized, got {:?}", result2);
    }

    #[tokio::test]
    async fn test_stream_detached_then_none() {
        // Detached is non-terminal — a new sender may reattach.
        // After Detached, the stream continues polling. Simulate no reattach
        // by sending Dropped, which IS terminal.
        let (tx, mut stream) = make_test_stream::<u32>();

        let bytes = rmp_serde::to_vec(&StreamFrame::<u32>::Detached).unwrap();
        tx.send(bytes).unwrap();

        let result = stream.next().await;
        assert!(matches!(result, Some(Ok(StreamFrame::Detached))), "expected Detached, got {:?}", result);

        // Send Dropped to signal no reattach — terminal sentinel.
        let bytes = rmp_serde::to_vec(&StreamFrame::<u32>::Dropped).unwrap();
        tx.send(bytes).unwrap();

        let result2 = stream.next().await;
        assert!(matches!(result2, Some(Err(StreamError::SenderDropped))), "expected SenderDropped after Detached, got {:?}", result2);

        let result3 = stream.next().await;
        assert!(result3.is_none(), "expected None after SenderDropped, got {:?}", result3);
    }

    #[tokio::test]
    async fn test_stream_dropped_then_none() {
        let (tx, mut stream) = make_test_stream::<u32>();

        let bytes = rmp_serde::to_vec(&StreamFrame::<u32>::Dropped).unwrap();
        tx.send(bytes).unwrap();

        let result = stream.next().await;
        match result {
            Some(Err(StreamError::SenderDropped)) => {}
            other => panic!("expected SenderDropped, got {:?}", other),
        }

        let result2 = stream.next().await;
        assert!(result2.is_none(), "expected None after Dropped, got {:?}", result2);
    }

    #[tokio::test]
    async fn test_stream_transport_error_then_none() {
        let (tx, mut stream) = make_test_stream::<u32>();

        let bytes = rmp_serde::to_vec(&StreamFrame::<u32>::TransportError("conn reset".to_string())).unwrap();
        tx.send(bytes).unwrap();

        let result = stream.next().await;
        match result {
            Some(Err(StreamError::TransportError(msg))) => assert_eq!(msg, "conn reset"),
            other => panic!("expected TransportError, got {:?}", other),
        }

        let result2 = stream.next().await;
        assert!(result2.is_none(), "expected None after TransportError, got {:?}", result2);
    }

    #[tokio::test]
    async fn test_stream_filters_heartbeat() {
        let (tx, mut stream) = make_test_stream::<u32>();

        // Send heartbeat then an item
        let hb_bytes = rmp_serde::to_vec(&StreamFrame::<u32>::Heartbeat).unwrap();
        tx.send(hb_bytes).unwrap();

        let item_bytes = rmp_serde::to_vec(&StreamFrame::Item(7u32)).unwrap();
        tx.send(item_bytes).unwrap();

        // Consumer should never see Heartbeat -- should get Item directly
        let result = stream.next().await;
        match result {
            Some(Ok(StreamFrame::Item(val))) => assert_eq!(val, 7),
            other => panic!("expected Item(7) (heartbeat filtered), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_stream_deserialization_error_then_none() {
        let (tx, mut stream) = make_test_stream::<u32>();

        // Send invalid bytes
        tx.send(vec![0xFF, 0xFE, 0xFD]).unwrap();

        let result = stream.next().await;
        match result {
            Some(Err(StreamError::DeserializationError(_))) => {}
            other => panic!("expected DeserializationError, got {:?}", other),
        }

        let result2 = stream.next().await;
        assert!(result2.is_none(), "expected None after DeserializationError, got {:?}", result2);
    }

    #[tokio::test]
    async fn test_stream_none_when_sender_dropped() {
        let mgr = make_manager();
        let mut stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        // Remove the anchor from the registry to drop the frame_tx sender,
        // then drop the returned entry so ALL senders are gone.
        let entry = mgr.remove_anchor(local_id);
        drop(entry); // drops frame_tx -> channel closes

        let result = stream.next().await;
        assert!(result.is_none(), "expected None when channel sender dropped, got {:?}", result);
    }

    #[tokio::test]
    async fn test_cancel_removes_anchor_from_registry() {
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        assert!(mgr.registry.contains_key(&local_id), "anchor must exist before cancel");

        // cancel(self) consumes the stream and removes anchor from registry
        stream.cancel();

        assert!(!mgr.registry.contains_key(&local_id), "anchor must be removed after cancel(self)");
    }

    // -----------------------------------------------------------------------
    // AnchorManagerBuilder + default_timeout tests (Plan 08-04, Task 1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_creates_manager_no_timeout() {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .build()
            .expect("builder with required fields should succeed");
        assert!(
            mgr.default_timeout.is_none(),
            "default_timeout must be None when not set"
        );
    }

    #[test]
    fn test_builder_creates_manager_with_timeout() {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("builder with timeout should succeed");
        assert_eq!(
            mgr.default_timeout,
            Some(std::time::Duration::from_secs(10)),
            "default_timeout must match configured value"
        );
    }

    #[test]
    fn test_convenience_new_still_works() {
        // AnchorManager::new must still compile and create a manager with no timeout
        let mgr = make_manager();
        assert!(
            mgr.default_timeout.is_none(),
            "AnchorManager::new must produce None default_timeout"
        );
    }

    #[tokio::test]
    async fn test_timeout_removes_unattached_anchor() {
        tokio::time::pause();

        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(1))
            .build()
            .expect("builder should succeed");

        let anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();
        let (_, local_id) = handle.unpack();

        assert!(mgr.registry.contains_key(&local_id), "anchor must exist after create");

        // Advance past the timeout
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        assert!(
            !mgr.registry.contains_key(&local_id),
            "anchor must be removed after timeout expires"
        );
    }

    #[tokio::test]
    async fn test_expired_anchor_returns_not_found() {
        tokio::time::pause();

        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(1))
            .build()
            .expect("builder should succeed");

        let anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();
        let (_, local_id) = handle.unpack();

        // Advance past timeout
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Try to attach -- should get AnchorNotFound
        let result = mgr.try_attach(local_id, handle);
        match result {
            Err(AttachError::AnchorNotFound { .. }) => {}
            other => panic!("expected AnchorNotFound after timeout, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_timeout_pauses_on_attach_resumes_on_detach() {
        tokio::time::pause();

        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("builder should succeed");

        let anchor = mgr.create_anchor::<u32>();
        let handle = anchor.handle();
        let (_, local_id) = handle.unpack();

        // Advance 1s (less than 2s timeout)
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert!(mgr.registry.contains_key(&local_id), "anchor must exist before timeout");

        // Attach -- should cancel the timeout task
        mgr.try_attach(local_id, handle).expect("attach should succeed");

        // Advance well past the original deadline
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        assert!(
            mgr.registry.contains_key(&local_id),
            "anchor must still exist while attached (timeout paused)"
        );

        // Detach -- should respawn the timeout task
        mgr.detach(local_id);

        // Advance past the new timeout (2s from detach)
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        assert!(
            !mgr.registry.contains_key(&local_id),
            "anchor must be removed after detach + timeout"
        );
    }

    // -----------------------------------------------------------------------
    // StreamAnchor::set_timeout tests (Plan 08-04, Task 2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_set_timeout_starts_timeout_on_no_default() {
        tokio::time::pause();

        // Manager with NO default timeout
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        // set_timeout starts a timeout task even though manager had no default
        stream.set_timeout(Some(std::time::Duration::from_secs(1)));

        assert!(mgr.registry.contains_key(&local_id), "anchor must exist before timeout");

        // Advance past the timeout
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        assert!(
            !mgr.registry.contains_key(&local_id),
            "anchor must be removed after set_timeout expires"
        );
    }

    #[tokio::test]
    async fn test_set_timeout_none_disables_timeout() {
        tokio::time::pause();

        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("builder should succeed");

        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        // Disable the timeout
        stream.set_timeout(None);

        // Advance well past the original deadline
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        assert!(
            mgr.registry.contains_key(&local_id),
            "anchor must still exist after disabling timeout"
        );
    }

    #[tokio::test]
    async fn test_set_timeout_overrides_default() {
        tokio::time::pause();

        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .default_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("builder should succeed");

        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        // Override with a shorter timeout
        stream.set_timeout(Some(std::time::Duration::from_secs(1)));

        // Advance 2s -- should trigger the 1s override, not the 10s default
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        assert!(
            !mgr.registry.contains_key(&local_id),
            "anchor must be removed by overridden 1s timeout, not waiting for 10s default"
        );
    }

    #[tokio::test]
    async fn test_set_timeout_while_attached_no_immediate_effect() {
        tokio::time::pause();

        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let handle = stream.handle();
        let (_, local_id) = handle.unpack();

        // Attach the anchor
        mgr.try_attach(local_id, handle).expect("attach should succeed");

        // Set a timeout while attached -- should NOT spawn a task immediately
        stream.set_timeout(Some(std::time::Duration::from_secs(1)));

        // Advance well past the timeout
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        // Anchor must still exist (attached, timeout only takes effect on detach)
        assert!(
            mgr.registry.contains_key(&local_id),
            "anchor must still exist while attached even with set_timeout"
        );

        // Detach -- now the timeout should kick in (stored duration from set_timeout)
        mgr.detach(local_id);

        // Advance past the timeout
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        assert!(
            !mgr.registry.contains_key(&local_id),
            "anchor must be removed after detach with stored set_timeout duration"
        );
    }

    // -----------------------------------------------------------------------
    // Registry injection tests (Plan 09-01, Task 2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_with_external_registry() {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let external_registry: Arc<DashMap<u64, AnchorEntry>> = Arc::new(DashMap::new());

        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .registry(external_registry.clone())
            .build()
            .expect("builder with external registry should succeed");

        // Verify the manager uses the injected registry (same Arc)
        assert!(Arc::ptr_eq(&mgr.registry, &external_registry),
            "manager must use the externally provided registry Arc");
    }

    #[test]
    fn test_builder_without_registry_creates_own() {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);

        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .build()
            .expect("builder without registry should succeed");

        // Registry should exist and be empty
        assert_eq!(mgr.registry.len(), 0, "auto-created registry must be empty");
    }

    #[test]
    fn test_create_anchor_inserts_into_shared_registry() {
        let worker_id = velo_common::WorkerId::from_u64(42);
        let transport: Arc<dyn crate::transport::FrameTransport> = Arc::new(MockTransport);
        let shared_registry: Arc<DashMap<u64, AnchorEntry>> = Arc::new(DashMap::new());

        let mgr = AnchorManagerBuilder::default()
            .worker_id(worker_id)
            .transport(transport)
            .registry(shared_registry.clone())
            .build()
            .expect("builder should succeed");

        assert_eq!(shared_registry.len(), 0, "shared registry must be empty before create_anchor");

        let anchor = mgr.create_anchor::<u32>();
        let (_, local_id) = anchor.handle().unpack();

        // Verify the entry was inserted into the shared registry (accessible outside mgr)
        assert_eq!(shared_registry.len(), 1, "shared registry must have 1 entry after create_anchor");
        assert!(shared_registry.contains_key(&local_id),
            "shared registry must contain the created anchor");
    }

    // -----------------------------------------------------------------------
    // StreamController tests (Plan 11-02, Task 1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_controller_clone() {
        // controller() returns a Clone-able type; multiple clones all refer to same anchor.
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        let ctrl1 = stream.controller();
        let ctrl2 = ctrl1.clone();

        // Both point to the same local_id — cancelling via ctrl2 removes the anchor.
        ctrl2.cancel();
        assert!(!mgr.registry.contains_key(&local_id), "ctrl2.cancel() must remove anchor from registry");

        // ctrl1 is now a no-op (AtomicBool already set), double-cancel must not panic.
        ctrl1.cancel();
    }

    #[test]
    fn test_cancel_self_removes_registry() {
        // StreamAnchor::cancel(self) removes anchor from registry.
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        assert!(mgr.registry.contains_key(&local_id), "anchor must exist before cancel");

        stream.cancel();

        assert!(!mgr.registry.contains_key(&local_id), "anchor must be removed after cancel(self)");
    }

    #[test]
    fn test_controller_cancel_removes_registry() {
        // StreamController::cancel() removes anchor from registry.
        // Test the drop path: get controller, drop StreamAnchor, verify controller still no-panics.
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        let ctrl = stream.controller();
        // Drop the stream — Drop impl fires, removes anchor via controller.cancel()
        drop(stream);

        assert!(!mgr.registry.contains_key(&local_id), "anchor must be removed by Drop");

        // ctrl.cancel() should be idempotent — anchor already gone, no panic.
        ctrl.cancel();
    }

    #[test]
    fn test_double_cancel_idempotent() {
        // cancel twice does not panic, registry entry absent after first cancel.
        let mgr = make_manager();
        let stream = mgr.create_anchor::<u32>();
        let (_, local_id) = stream.handle().unpack();

        let ctrl = stream.controller();
        ctrl.cancel();
        assert!(!mgr.registry.contains_key(&local_id), "anchor must be absent after first cancel");

        // Second cancel — must not panic.
        ctrl.cancel();
    }

    // -----------------------------------------------------------------------
    // register_handlers tests (Plan 12-01, Task 2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_handlers_stores_messenger_in_lock() {
        // Verify that after register_handlers, messenger_lock.get() is Some
        // and that a second call returns Err.
        // Note: We use Messenger::builder().build() which requires tokio runtime.
        // This is a compile + behavior test using a real Messenger (no-transport).
        // The test is sync to avoid needing #[tokio::test] but uses a runtime.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let messenger = velo_messenger::Messenger::builder()
                .build()
                .await
                .expect("messenger");
            let worker_id = velo_common::WorkerId::from_u64(99);
            let transport = Arc::new(MockTransport);
            let am = Arc::new(AnchorManager::new(worker_id, transport));

            // First call succeeds
            am.register_handlers(Arc::clone(&messenger))
                .expect("first register_handlers must succeed");

            // messenger_lock is set
            assert!(am.messenger_lock.get().is_some(), "messenger_lock must be Some after register_handlers");
        });
    }

    #[test]
    fn test_register_handlers_second_call_errors() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let m1 = velo_messenger::Messenger::builder().build().await.unwrap();
            let m2 = velo_messenger::Messenger::builder().build().await.unwrap();

            let worker_id = velo_common::WorkerId::from_u64(100);
            let transport = Arc::new(MockTransport);
            let am = Arc::new(AnchorManager::new(worker_id, transport));

            am.register_handlers(Arc::clone(&m1)).expect("first call ok");
            let result = am.register_handlers(Arc::clone(&m2));
            assert!(result.is_err(), "second call must return Err");
        });
    }
}
