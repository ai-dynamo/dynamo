// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NovaStreaming - the main entry point for stream-based communication.
//!
//! `NovaStreaming` wraps a Nova instance and provides stream creation and
//! handle arming functionality. It registers the necessary handlers for
//! stream attach/cancel RPCs and frame delivery.
//!
//! # Example
//!
//! ```ignore
//! // Create NovaStreaming from Nova
//! let streaming = NovaStreaming::builder(nova.clone()).build()?;
//!
//! // Receiver side: create a stream
//! let (receiver, handle) = streaming.create_stream::<MyData>();
//! send_handle_to_sender(handle);
//!
//! // Sender side: arm and attach
//! let handle = receive_handle();
//! let sender = streaming.arm::<MyData>(handle).attach().await?;
//! sender.send(data).await?;
//! sender.finalize().await?;
//! ```

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_identity::InstanceId;
use dynamo_nova::am::NovaHandler;
use dynamo_nova::Nova;
use futures::Stream;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::Notify;
use tokio::task::AbortHandle;
use uuid::Uuid;

use crate::channel::{AnchorRegistry, StreamReceiver, StreamReceiverBuilder};
use crate::error::{AttachError, FinalizeError, SendError, TrySendError};
use crate::protocol::{
    AnchorId, AttachRequest, AttachResponse, CancelAck, CancelRequest, RawWireFrame, StreamFrame,
};

/// Minimal handle for wire transport - just the anchor ID.
///
/// This is untyped on the wire; the type `T` is provided when arming via
/// `NovaStreaming::arm::<T>()`. If the sender's type doesn't match the
/// receiver's expected type, deserialization will fail and an error frame
/// will be sent back to the sender.
pub type StreamHandle = AnchorId;

/// Registry that maps anchor IDs to stream receivers.
///
/// This allows incoming frames to be routed to the correct receiver.
#[derive(Clone)]
pub struct ReceiverRegistry {
    /// Type-erased receivers - we store the frame sender channel.
    receivers: Arc<DashMap<u128, ReceiverEntry>>,
}

/// Attached session info for tracking who's connected.
#[derive(Clone)]
struct AttachedSessionInfo {
    sender_instance: InstanceId,
    session_id: uuid::Uuid,
}

/// Entry in the receiver registry.
struct ReceiverEntry {
    /// Channel to send frames to the receiver.
    /// We use serde_json::Value for type erasure.
    frame_sender: flume::Sender<serde_json::Value>,
    /// The receiver's message timeout (for heartbeat interval calculation).
    message_timeout: Option<Duration>,
    /// Current attached session (if any).
    attached_session: parking_lot::Mutex<Option<AttachedSessionInfo>>,
    /// Whether the receiver has been cancelled.
    cancelled: std::sync::atomic::AtomicBool,
    /// Callback for liveness checks.
    is_alive: Box<dyn Fn() -> bool + Send + Sync>,
    /// Callback for on_attach.
    on_attach: Box<dyn Fn(InstanceId, uuid::Uuid) -> bool + Send + Sync>,
    /// Callback for on_detach.
    on_detach: Box<dyn Fn(uuid::Uuid) -> bool + Send + Sync>,
    /// Callback for on_message_received.
    on_message_received: Box<dyn Fn() -> bool + Send + Sync>,
    /// Callback for on_cancel.
    on_cancel: Box<dyn Fn(String) -> bool + Send + Sync>,
}

impl ReceiverRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            receivers: Arc::new(DashMap::new()),
        }
    }

    /// Register a receiver for an anchor.
    pub(crate) fn register<T: DeserializeOwned + Send + 'static>(
        &self,
        receiver: &StreamReceiver<T>,
    ) {
        let anchor_id = receiver.anchor_id();
        let frame_sender = receiver.frame_sender();
        let message_timeout = receiver.message_timeout();
        let context = receiver.context();
        let context_weak = context.downgrade();

        let registry = self.clone();
        let _ = receiver.set_drop_hook(Arc::new(move |anchor_id| {
            registry.unregister(anchor_id);
        }));

        // Create a type-erased sender that deserializes and forwards
        let (tx, rx) = flume::bounded::<serde_json::Value>(256);

        // Spawn a task to deserialize and forward frames
        let typed_sender = frame_sender.clone();
        tokio::spawn(async move {
            while let Ok(value) = rx.recv_async().await {
                match serde_json::from_value::<StreamFrame<T>>(value) {
                    Ok(frame) => {
                        // Try non-blocking first, fall back to async wait if full
                        match typed_sender.try_send(frame) {
                            Ok(()) => {}
                            Err(flume::TrySendError::Full(f)) => {
                                // Channel full - wait for space
                                if typed_sender.send_async(f).await.is_err() {
                                    break;
                                }
                            }
                            Err(flume::TrySendError::Disconnected(_)) => break,
                        }
                    }
                    Err(e) => {
                        // Type mismatch - send error frame
                        let _ = typed_sender.try_send(StreamFrame::Error(format!(
                            "deserialization error: {}",
                            e
                        )));
                        break;
                    }
                }
            }
        });

        let receiver_alive = context_weak.clone();
        let receiver_attach = context_weak.clone();
        let receiver_detach = context_weak.clone();
        let receiver_message = context_weak.clone();
        let receiver_cancel = context_weak;

        let entry = ReceiverEntry {
            frame_sender: tx,
            message_timeout,
            attached_session: parking_lot::Mutex::new(None),
            cancelled: std::sync::atomic::AtomicBool::new(false),
            is_alive: Box::new(move || receiver_alive.upgrade().is_some()),
            on_attach: Box::new(move |instance, session| {
                receiver_attach
                    .upgrade()
                    .map(|receiver| receiver.on_attach(instance, session))
                    .unwrap_or(false)
            }),
            on_detach: Box::new(move |session| {
                receiver_detach
                    .upgrade()
                    .map(|receiver| receiver.on_detach(session))
                    .unwrap_or(false)
            }),
            on_message_received: Box::new(move || {
                receiver_message
                    .upgrade()
                    .map(|receiver| receiver.on_message_received())
                    .unwrap_or(false)
            }),
            on_cancel: Box::new(move |reason| {
                if let Some(receiver) = receiver_cancel.upgrade() {
                    let _ = receiver.cancel(reason);
                    true
                } else {
                    false
                }
            }),
        };

        self.receivers.insert(anchor_id.as_u128(), entry);
    }

    /// Unregister a receiver.
    pub fn unregister(&self, anchor_id: AnchorId) {
        self.receivers.remove(&anchor_id.as_u128());
    }

    /// Deliver a frame to the receiver for an anchor.
    ///
    /// Returns Ok(()) if delivered, Err if receiver not found or channel closed.
    /// Frames are silently dropped if the receiver has been cancelled.
    pub fn deliver(&self, anchor_id: AnchorId, frame_value: serde_json::Value) -> Result<()> {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let result = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                Ok(())
            } else if entry.cancelled.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(()) // Silently drop - receiver is shutting down
            } else if !(entry.on_message_received)() {
                remove = true;
                Ok(())
            } else {
                entry
                    .frame_sender
                    .try_send(frame_value)
                    .map_err(|_| anyhow::anyhow!("receiver channel full or closed"))?;
                Ok(())
            }
        } else {
            Ok(())
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        result
    }

    /// Called when a sender attaches to an anchor.
    ///
    /// Returns `None` if the receiver has been cancelled (attach rejected).
    /// Returns `Some(timeout)` on success, where timeout may be `None` if not configured.
    pub fn on_attach(&self, anchor_id: AnchorId, sender_instance: InstanceId, session_id: uuid::Uuid) -> Option<Option<Duration>> {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let result = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                None
            } else if entry.cancelled.load(std::sync::atomic::Ordering::SeqCst)
                || !(entry.on_attach)(sender_instance, session_id)
            {
                // Reject attach but keep receiver registered
                None
            } else {
                *entry.attached_session.lock() = Some(AttachedSessionInfo {
                    sender_instance,
                    session_id,
                });
                Some(entry.message_timeout)
            }
        } else {
            None
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        result
    }

    /// Called when a sender detaches from an anchor.
    pub fn on_detach(&self, anchor_id: AnchorId, session_id: uuid::Uuid) {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() || !(entry.on_detach)(session_id) {
                remove = true;
            } else {
                *entry.attached_session.lock() = None;
            }
        }

        if remove {
            self.receivers.remove(&anchor_key);
        }
    }

    /// Get the message timeout for a receiver (used to calculate heartbeat interval).
    pub fn get_message_timeout(&self, anchor_id: AnchorId) -> Option<Duration> {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let timeout = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                None
            } else {
                entry.message_timeout
            }
        } else {
            None
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        timeout
    }

    /// Check if a receiver has been cancelled.
    pub fn is_cancelled(&self, anchor_id: AnchorId) -> bool {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let cancelled = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                true
            } else {
                entry.cancelled.load(std::sync::atomic::Ordering::SeqCst)
            }
        } else {
            true
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        cancelled
    }

    /// Check if a receiver exists and is available for attach.
    pub fn is_available(&self, anchor_id: AnchorId) -> bool {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let available = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                false
            } else {
                !entry.cancelled.load(std::sync::atomic::Ordering::SeqCst)
            }
        } else {
            false
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        available
    }

    /// Cancel a stream by anchor ID.
    ///
    /// This method:
    /// 1. Marks the receiver as cancelled (rejects future attach attempts)
    /// 2. Delivers a Cancelled frame to the receiver
    /// 3. Returns the attached session info (if any) so caller can notify the sender
    /// 4. Does NOT unregister - caller should do that after notifying sender
    ///
    /// Returns `Some((sender_instance, session_id))` if a sender was attached.
    pub fn cancel(&self, anchor_id: AnchorId, reason: String) -> Option<(InstanceId, uuid::Uuid)> {
        let anchor_key = anchor_id.as_u128();
        let mut remove = false;

        let attached = if let Some(entry) = self.receivers.get(&anchor_key) {
            if !(entry.is_alive)() {
                remove = true;
                None
            } else {
                // Mark as cancelled first (rejects new attach attempts)
                entry.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);

                // Call the cancel callback to notify the receiver
                if !(entry.on_cancel)(reason) {
                    remove = true;
                }

                // Take and return the attached session info
                entry
                    .attached_session
                    .lock()
                    .take()
                    .map(|s| (s.sender_instance, s.session_id))
            }
        } else {
            None
        };

        if remove {
            self.receivers.remove(&anchor_key);
        }

        attached
    }
}

impl Default for ReceiverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SenderRegistry - Track active senders for cancellation
// =============================================================================

/// Cancel reason slot shared between sender and registry.
type CancelReasonSlot = Arc<parking_lot::Mutex<Option<String>>>;

/// Registry for active stream senders.
///
/// Maps (anchor_id, session_id) to the sender's cancel_reason slot,
/// allowing cancel handlers to notify senders of cancellation.
#[derive(Clone)]
pub(crate) struct SenderRegistry {
    senders: Arc<DashMap<(u128, Uuid), CancelReasonSlot>>,
}

impl SenderRegistry {
    /// Create a new empty sender registry.
    pub fn new() -> Self {
        Self {
            senders: Arc::new(DashMap::new()),
        }
    }

    /// Register a sender's cancel_reason slot.
    pub fn register(
        &self,
        anchor_id: AnchorId,
        session_id: Uuid,
        cancel_reason: CancelReasonSlot,
    ) {
        self.senders
            .insert((anchor_id.as_u128(), session_id), cancel_reason);
    }

    /// Unregister a sender.
    pub fn unregister(&self, anchor_id: AnchorId, session_id: Uuid) {
        self.senders.remove(&(anchor_id.as_u128(), session_id));
    }

    /// Cancel a sender (set its cancel_reason).
    ///
    /// Returns true if the sender was found and cancelled.
    pub fn cancel(&self, anchor_id: AnchorId, session_id: Uuid, reason: String) -> bool {
        if let Some(entry) = self.senders.get(&(anchor_id.as_u128(), session_id)) {
            *entry.lock() = Some(reason);
            true
        } else {
            false
        }
    }
}

// =============================================================================
// NovaAmSink - Internal sink for sending frames via Nova AM
// =============================================================================

/// Error slot for capturing background task errors.
struct SinkErrorSlot {
    error: parking_lot::Mutex<Option<anyhow::Error>>,
}

impl SinkErrorSlot {
    fn new() -> Self {
        Self {
            error: parking_lot::Mutex::new(None),
        }
    }

    fn set_error(&self, err: anyhow::Error) {
        let mut guard = self.error.lock();
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    fn take_error(&self) -> Option<anyhow::Error> {
        self.error.lock().take()
    }
}

/// Internal sink for sending frames via Nova AM.
///
/// This sink buffers frames and sends them to the target instance via Nova's
/// active message infrastructure.
pub(crate) struct NovaAmSink {
    anchor_id: AnchorId,
    tx: flume::Sender<Bytes>,
    error_slot: Arc<SinkErrorSlot>,
    closed: Arc<AtomicBool>,
}

impl NovaAmSink {
    /// Create a new sink that sends frames to the target instance.
    pub fn new(
        nova: Arc<Nova>,
        anchor_id: AnchorId,
        target_instance: InstanceId,
        capacity: usize,
    ) -> Self {
        let (tx, rx) = flume::bounded(capacity);
        let error_slot = Arc::new(SinkErrorSlot::new());

        // Spawn background task to send frames via Nova AM
        let error_slot_clone = Arc::clone(&error_slot);
        tokio::spawn(async move {
            while let Ok(payload) = rx.recv_async().await {
                let result = nova
                    .am_send(STREAM_DATA_HANDLER)
                    .map(|b| b.instance(target_instance).raw_payload(payload).send());

                match result {
                    Ok(fut) => {
                        if let Err(e) = fut.await {
                            error_slot_clone.set_error(e);
                            break;
                        }
                    }
                    Err(e) => {
                        error_slot_clone.set_error(e);
                        break;
                    }
                }
            }
        });

        Self {
            anchor_id,
            tx,
            error_slot,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Try to send a frame without blocking (hot path).
    pub fn try_send<T: Serialize>(
        &self,
        frame: StreamFrame<T>,
    ) -> Result<(), TrySendError<StreamFrame<T>>> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(TrySendError::Closed);
        }

        if let Some(err) = self.error_slot.take_error() {
            return Err(TrySendError::BackgroundError(err));
        }

        // Check capacity before serializing
        if self.tx.is_full() {
            return Err(TrySendError::Full(frame));
        }

        // Wrap frame with anchor_id for routing
        let wire_frame = RawWireFrame {
            anchor_id: self.anchor_id,
            frame: serde_json::to_value(&frame)
                .map_err(|e| TrySendError::Serialization(e.to_string()))?,
        };

        let payload = serde_json::to_vec(&wire_frame)
            .map_err(|e| TrySendError::Serialization(e.to_string()))?;

        match self.tx.try_send(Bytes::from(payload)) {
            Ok(()) => Ok(()),
            Err(flume::TrySendError::Full(_)) => Err(TrySendError::Full(frame)),
            Err(flume::TrySendError::Disconnected(_)) => Err(TrySendError::Closed),
        }
    }

    /// Send a frame asynchronously (waits if buffer is full).
    pub async fn send<T: Serialize>(&self, frame: StreamFrame<T>) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            anyhow::bail!("sink closed");
        }

        // Wrap frame with anchor_id for routing
        let wire_frame = RawWireFrame {
            anchor_id: self.anchor_id,
            frame: serde_json::to_value(&frame)?,
        };

        let payload = Bytes::from(serde_json::to_vec(&wire_frame)?);

        self.tx
            .send_async(payload)
            .await
            .map_err(|_| anyhow::anyhow!("sink closed"))?;

        if let Some(err) = self.error_slot.take_error() {
            return Err(err);
        }

        Ok(())
    }

    /// Check for background errors without consuming them.
    pub fn poll_error(&self) -> Result<()> {
        if let Some(err) = self.error_slot.take_error() {
            return Err(err);
        }
        Ok(())
    }

    /// Close the sink.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}

// =============================================================================
// StreamSender - Send data to a stream receiver
// =============================================================================

/// Stream sender for sending data to a receiver.
///
/// Created by `ArmedHandle::attach()`. Use `send()` or `try_send()` to send
/// data, and `finalize()` when done.
pub struct StreamSender<T> {
    anchor_id: AnchorId,
    target_instance: InstanceId,
    session_id: Uuid,
    sink: NovaAmSink,
    heartbeat_interval: Option<Duration>,
    send_notify: Arc<Notify>,
    heartbeat_abort: Option<AbortHandle>,
    cancel_reason: Arc<parking_lot::Mutex<Option<String>>>,
    sender_registry: SenderRegistry,
    finalized: AtomicBool,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> StreamSender<T>
where
    T: Serialize + Send + 'static,
{
    /// Create a new StreamSender.
    pub(crate) fn new(
        anchor_id: AnchorId,
        target_instance: InstanceId,
        session_id: Uuid,
        sink: NovaAmSink,
        heartbeat_interval: Option<Duration>,
        sender_registry: SenderRegistry,
    ) -> Self {
        let send_notify = Arc::new(Notify::new());
        let cancel_reason = Arc::new(parking_lot::Mutex::new(None));

        // Register in sender registry so cancel RPCs can find us
        sender_registry.register(anchor_id, session_id, Arc::clone(&cancel_reason));

        // Spawn heartbeat task if interval is configured
        let heartbeat_abort = heartbeat_interval.map(|interval| {
            let sink_anchor = sink.anchor_id;
            let sink_tx = sink.tx.clone();
            let sink_closed = Arc::clone(&sink.closed);
            let send_notify_clone = Arc::clone(&send_notify);

            let handle = tokio::spawn(heartbeat_task::<T>(
                sink_anchor,
                sink_tx,
                sink_closed,
                send_notify_clone,
                interval,
            ));
            handle.abort_handle()
        });

        Self {
            anchor_id,
            target_instance,
            session_id,
            sink,
            heartbeat_interval,
            send_notify,
            heartbeat_abort,
            cancel_reason,
            sender_registry,
            finalized: AtomicBool::new(false),
            _phantom: PhantomData,
        }
    }

    /// Get the anchor ID.
    pub fn anchor_id(&self) -> AnchorId {
        self.anchor_id
    }

    /// Get the target instance ID.
    pub fn target_instance(&self) -> InstanceId {
        self.target_instance
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Get the heartbeat interval.
    pub fn heartbeat_interval(&self) -> Option<Duration> {
        self.heartbeat_interval
    }

    /// Check if the stream has been cancelled by the receiver.
    pub fn check_cancelled(&self) -> Option<String> {
        self.cancel_reason.lock().clone()
    }

    /// Try to send an item without blocking (hot path).
    ///
    /// Returns `Ok(())` if queued, `Err(TrySendError::Full(item))` if buffer is full.
    pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
        self.try_send_frame(StreamFrame::Data(item))
            .map_err(|e| match e {
                TrySendError::Full(StreamFrame::Data(data)) => TrySendError::Full(data),
                TrySendError::Full(_) => unreachable!("always Data frame"),
                TrySendError::BackgroundError(e) => TrySendError::BackgroundError(e),
                TrySendError::Closed => TrySendError::Closed,
                TrySendError::Cancelled(r) => TrySendError::Cancelled(r),
                TrySendError::Serialization(e) => TrySendError::Serialization(e),
            })
    }

    /// Send an item asynchronously (waits if buffer is full).
    pub async fn send(&self, item: T) -> Result<(), SendError> {
        self.send_frame(StreamFrame::Data(item)).await
    }

    /// Send raw pre-serialized bytes.
    pub async fn send_raw(&self, bytes: impl Into<Vec<u8>>) -> Result<(), SendError> {
        self.send_frame(StreamFrame::RawData(bytes.into())).await
    }

    /// Try to send raw bytes without blocking.
    pub fn try_send_raw(&self, bytes: Vec<u8>) -> Result<(), TrySendError<Vec<u8>>> {
        self.try_send_frame(StreamFrame::RawData(bytes))
            .map_err(|e| match e {
                TrySendError::Full(StreamFrame::RawData(data)) => TrySendError::Full(data),
                TrySendError::Full(_) => unreachable!("always RawData frame"),
                TrySendError::BackgroundError(e) => TrySendError::BackgroundError(e),
                TrySendError::Closed => TrySendError::Closed,
                TrySendError::Cancelled(r) => TrySendError::Cancelled(r),
                TrySendError::Serialization(e) => TrySendError::Serialization(e),
            })
    }

    /// Poll for progress and check for background errors.
    pub fn progress(&self) -> Result<(), SendError> {
        if self.cancel_reason.lock().is_some() {
            return Err(SendError::Cancelled);
        }

        self.sink.poll_error().map_err(SendError::Transport)?;
        Ok(())
    }

    /// Forward all items from a stream.
    pub async fn forward<S>(&self, stream: S) -> Result<(), SendError>
    where
        S: Stream<Item = T> + Send,
    {
        use futures::StreamExt;
        let mut stream = std::pin::pin!(stream);
        while let Some(item) = stream.next().await {
            self.send(item).await?;
        }
        Ok(())
    }

    /// Finalize the stream, signaling no more data will be sent.
    pub async fn finalize(self) -> Result<(), FinalizeError> {
        if self.finalized.swap(true, Ordering::SeqCst) {
            return Err(FinalizeError::AlreadyFinalized);
        }

        // Abort heartbeat task
        if let Some(abort) = &self.heartbeat_abort {
            abort.abort();
        }

        // Send finalize frame
        self.sink
            .send(StreamFrame::<T>::Finalized)
            .await
            .map_err(FinalizeError::Failed)?;

        // Close the sink
        self.sink.close();

        Ok(())
    }

    /// Convert to a channel-like interface for fine-grained control.
    pub fn send_mode(self) -> SendMode<T> {
        SendMode {
            sender: Arc::new(self),
        }
    }

    /// Try to send a frame without blocking (internal).
    fn try_send_frame(&self, frame: StreamFrame<T>) -> Result<(), TrySendError<StreamFrame<T>>> {
        if let Some(reason) = self.cancel_reason.lock().clone() {
            return Err(TrySendError::Cancelled(reason));
        }

        if self.finalized.load(Ordering::SeqCst) {
            return Err(TrySendError::Closed);
        }

        let result = self.sink.try_send(frame);
        if result.is_ok() {
            self.send_notify.notify_one();
        }
        result
    }

    /// Send a frame asynchronously (internal).
    async fn send_frame(&self, frame: StreamFrame<T>) -> Result<(), SendError> {
        if self.cancel_reason.lock().is_some() {
            return Err(SendError::Cancelled);
        }

        if self.finalized.load(Ordering::SeqCst) {
            return Err(SendError::Closed);
        }

        self.sink
            .send(frame)
            .await
            .map_err(SendError::Transport)?;

        self.send_notify.notify_one();
        Ok(())
    }
}

impl<T> Drop for StreamSender<T> {
    fn drop(&mut self) {
        // Unregister from sender registry
        self.sender_registry
            .unregister(self.anchor_id, self.session_id);

        // Abort heartbeat task
        if let Some(abort) = &self.heartbeat_abort {
            abort.abort();
        }

        // Send finalized frame if not already finalized
        if !self.finalized.swap(true, Ordering::SeqCst) {
            let anchor_id = self.anchor_id;
            let tx = self.sink.tx.clone();

            // Best effort - fire and forget
            let wire_frame = RawWireFrame {
                anchor_id,
                frame: serde_json::to_value(StreamFrame::<()>::Finalized).unwrap_or_default(),
            };

            if let Ok(payload) = serde_json::to_vec(&wire_frame) {
                let _ = tx.try_send(Bytes::from(payload));
            }
        }
    }
}

// =============================================================================
// SendMode - Channel-like interface for sending
// =============================================================================

/// Channel-like interface for sending items with fine-grained control.
///
/// Use `try_send` for the hot path to avoid heap allocations.
pub struct SendMode<T>
where
    T: Serialize + Send + 'static,
{
    sender: Arc<StreamSender<T>>,
}

impl<T> SendMode<T>
where
    T: Serialize + Send + 'static,
{
    /// Get the anchor ID.
    pub fn anchor_id(&self) -> AnchorId {
        self.sender.anchor_id()
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Uuid {
        self.sender.session_id()
    }

    /// Try to send an item without blocking (hot path).
    pub fn try_send(&self, item: T) -> Result<(), TrySendError<T>> {
        self.sender.try_send(item)
    }

    /// Poll for progress and check for background errors.
    pub fn progress(&self) -> Result<(), SendError> {
        self.sender.progress()
    }

    /// Send an item asynchronously.
    pub async fn send(&self, item: T) -> Result<(), SendError> {
        self.sender.send(item).await
    }

    /// Send raw pre-serialized bytes.
    pub async fn send_raw(&self, bytes: impl Into<Vec<u8>>) -> Result<(), SendError> {
        self.sender.send_raw(bytes).await
    }

    /// Try to send raw bytes without blocking.
    pub fn try_send_raw(&self, bytes: Vec<u8>) -> Result<(), TrySendError<Vec<u8>>> {
        self.sender.try_send_raw(bytes)
    }

    /// Finalize the stream.
    pub async fn finalize(self) -> Result<(), FinalizeError> {
        match Arc::try_unwrap(self.sender) {
            Ok(sender) => sender.finalize().await,
            Err(_) => Err(FinalizeError::Failed(anyhow::anyhow!(
                "cannot finalize: other references exist"
            ))),
        }
    }
}

impl<T> Clone for SendMode<T>
where
    T: Serialize + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            sender: Arc::clone(&self.sender),
        }
    }
}

/// Background heartbeat task.
async fn heartbeat_task<T: Serialize + Send + 'static>(
    anchor_id: AnchorId,
    tx: flume::Sender<Bytes>,
    closed: Arc<AtomicBool>,
    send_notify: Arc<Notify>,
    interval: Duration,
) {
    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                if closed.load(Ordering::SeqCst) {
                    return;
                }

                // Send heartbeat frame
                let wire_frame = RawWireFrame {
                    anchor_id,
                    frame: serde_json::to_value(StreamFrame::<T>::Heartbeat).unwrap_or_default(),
                };

                if let Ok(payload) = serde_json::to_vec(&wire_frame)
                    && tx.try_send(Bytes::from(payload)).is_err()
                {
                    return; // Channel closed
                }
            }
            _ = send_notify.notified() => {
                // A message was sent, restart the timer
                continue;
            }
        }
    }
}

/// The main entry point for stream-based communication over Nova.
///
/// `NovaStreaming` wraps a Nova instance and handles:
/// - Stream creation (receivers)
/// - Handle arming (for senders)
/// - RPC handler registration for attach/cancel
/// - Frame delivery via AM
pub struct NovaStreaming {
    nova: Arc<Nova>,
    registry: ReceiverRegistry,
    sender_registry: SenderRegistry,
    anchor_registry: Arc<AnchorRegistry>,
    buffer_capacity: usize,
}

impl NovaStreaming {
    /// Create a builder for NovaStreaming.
    pub fn builder(nova: Arc<Nova>) -> NovaStreamingBuilder {
        NovaStreamingBuilder::new(nova)
    }

    /// Create a new stream receiver with default configuration.
    ///
    /// Returns the receiver and an untyped handle to send to remote senders.
    /// The handle is just an `AnchorId` - truly minimal on the wire.
    pub fn create_stream<T>(&self) -> (StreamReceiver<T>, StreamHandle)
    where
        T: DeserializeOwned + Send + 'static,
    {
        self.create_stream_with(|b| b)
    }

    /// Create a new stream receiver with custom configuration.
    ///
    /// The configure function receives a builder and can set options like
    /// attach_timeout, message_timeout, and capacity.
    pub fn create_stream_with<T, F>(&self, configure: F) -> (StreamReceiver<T>, StreamHandle)
    where
        T: DeserializeOwned + Send + 'static,
        F: FnOnce(StreamReceiverBuilder) -> StreamReceiverBuilder,
    {
        let instance_id = self.nova.instance_id();
        let endpoint = format!("nova://{}", instance_id);

        let builder = StreamReceiverBuilder::new(
            instance_id,
            endpoint,
            Arc::clone(&self.anchor_registry),
        );
        let builder = configure(builder);
        let receiver: StreamReceiver<T> = builder.build();

        let handle = receiver.anchor_id();

        // Register in our registry for frame delivery
        self.registry.register(&receiver);

        (receiver, handle)
    }

    /// Arm a handle for attachment.
    ///
    /// Type `T` is the type you intend to send. If this doesn't match the
    /// receiver's expected type, deserialization will fail on the receiver
    /// side and an error frame will be sent back.
    pub fn arm<T>(&self, handle: StreamHandle) -> ArmedHandle<'_, T>
    where
        T: Serialize + Send + 'static,
    {
        ArmedHandle {
            streaming: self,
            anchor_id: handle,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying Nova instance.
    pub fn nova(&self) -> &Arc<Nova> {
        &self.nova
    }

    /// Cancel a stream by its handle.
    ///
    /// This method is called by the receiver to cancel a stream. It:
    /// 1. Marks the receiver as cancelled (rejects future attach attempts)
    /// 2. Delivers a `Cancelled` frame to the receiver
    /// 3. If a sender is attached, sends a cancel notification (best effort)
    /// 4. Unregisters the receiver from the registry
    ///
    /// The cancel notification to the sender is fire-and-forget. If the sender
    /// has gone away, the cancel simply cleans up locally.
    pub async fn cancel_stream(&self, handle: StreamHandle, reason: impl Into<String>) {
        let reason = reason.into();
        let anchor_id = handle;

        // Cancel in registry and get attached session if any
        let attached = self.registry.cancel(anchor_id, reason.clone());

        // If a sender was attached, try to notify them (best effort)
        if let Some((sender_instance, session_id)) = attached {
            // Fire-and-forget cancel notification to sender
            // We don't wait for ack - if sender is gone, that's fine
            let nova = self.nova.clone();
            let cancel_reason = reason.clone();
            tokio::spawn(async move {
                let result = nova
                    .typed_unary::<CancelAck>(STREAM_CANCEL_HANDLER)
                    .ok()
                    .and_then(|b| {
                        b.payload(&CancelRequest {
                            anchor_id,
                            session_id,
                            reason: cancel_reason,
                        })
                        .ok()
                    })
                    .map(|b| b.instance(sender_instance).send());

                if let Some(fut) = result {
                    // Best effort - ignore errors (sender may be gone)
                    let _ = fut.await;
                }
            });
        }

        // Unregister from the registry
        self.registry.unregister(anchor_id);
    }
}

/// Builder for creating `NovaStreaming` instances.
pub struct NovaStreamingBuilder {
    nova: Arc<Nova>,
    buffer_capacity: usize,
}

impl NovaStreamingBuilder {
    fn new(nova: Arc<Nova>) -> Self {
        Self {
            nova,
            buffer_capacity: 256,
        }
    }

    /// Set the buffer capacity for send queues.
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Build the `NovaStreaming` instance.
    ///
    /// This registers the necessary Nova handlers for stream operations.
    pub fn build(self) -> Result<NovaStreaming> {
        let registry = ReceiverRegistry::new();
        let sender_registry = SenderRegistry::new();
        let anchor_registry = Arc::new(AnchorRegistry::new(self.nova.instance_id().worker_id()));

        // Register Nova handlers for stream operations
        register_stream_handlers(&self.nova, registry.clone(), sender_registry.clone())?;

        let streaming = NovaStreaming {
            nova: self.nova,
            registry,
            sender_registry,
            anchor_registry,
            buffer_capacity: self.buffer_capacity,
        };

        Ok(streaming)
    }
}

/// Armed handle ready for attachment.
///
/// Created by `NovaStreaming::arm()`. Call `attach()` to connect to the
/// receiver and start sending data.
pub struct ArmedHandle<'a, T> {
    streaming: &'a NovaStreaming,
    anchor_id: AnchorId,
    _phantom: PhantomData<fn() -> T>,
}

impl<'a, T> ArmedHandle<'a, T>
where
    T: Serialize + Send + 'static,
{
    /// Get the anchor ID.
    pub fn anchor_id(&self) -> AnchorId {
        self.anchor_id
    }

    /// Attach to the anchor and begin streaming.
    ///
    /// This sends an attach RPC to the anchor owner, negotiates configuration,
    /// and returns a `StreamSender` for sending data.
    /// Attach to the stream and start sending data.
    ///
    /// This sends an attach RPC to the anchor owner, negotiates configuration,
    /// and returns a `StreamSender` for sending data.
    pub async fn attach(self) -> Result<StreamSender<T>, AttachError> {
        let owner_worker = self.anchor_id.owner_worker();
        let session_id = Uuid::new_v4();
        let sender_instance = self.streaming.nova.instance_id();

        // Send attach RPC to the anchor owner
        // The .worker() method will handle discovery if needed
        let response: AttachResponse = self
            .streaming
            .nova
            .typed_unary::<AttachResponse>(STREAM_ATTACH_HANDLER)
            .map_err(AttachError::ConnectionFailed)?
            .payload(&AttachRequest {
                anchor_id: self.anchor_id,
                session_id,
                sender_instance,
            })
            .map_err(AttachError::ConnectionFailed)?
            .worker(owner_worker)
            .send()
            .await
            .map_err(AttachError::ConnectionFailed)?;

        // Parse target instance from endpoint
        let target_instance = parse_endpoint_instance(&response.stream_endpoint)
            .map_err(AttachError::ConnectionFailed)?;

        // Create sink with configured buffer capacity
        let heartbeat_interval = response.heartbeat_interval_ms.map(Duration::from_millis);
        let sink = NovaAmSink::new(
            Arc::clone(&self.streaming.nova),
            self.anchor_id,
            target_instance,
            self.streaming.buffer_capacity,
        );

        Ok(StreamSender::new(
            self.anchor_id,
            target_instance,
            session_id,
            sink,
            heartbeat_interval,
            self.streaming.sender_registry.clone(),
        ))
    }
}

/// Parse instance ID from endpoint string.
fn parse_endpoint_instance(endpoint: &str) -> Result<InstanceId> {
    // Format: nova://{instance_id}
    let stripped = endpoint
        .strip_prefix("nova://")
        .ok_or_else(|| anyhow::anyhow!("invalid endpoint format: {}", endpoint))?;

    let uuid: uuid::Uuid = stripped
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid instance ID in endpoint: {}", stripped))?;

    Ok(InstanceId::from(uuid))
}

// =============================================================================
// Nova Handler Registration
// =============================================================================

/// Handler names for stream operations.
pub const STREAM_ATTACH_HANDLER: &str = "stream_attach";
pub const STREAM_CANCEL_HANDLER: &str = "stream_cancel";
pub const STREAM_DATA_HANDLER: &str = "stream_data";

/// Register all stream handlers on a Nova instance.
fn register_stream_handlers(
    nova: &Arc<Nova>,
    registry: ReceiverRegistry,
    sender_registry: SenderRegistry,
) -> Result<()> {
    // Clone registries for each handler
    let registry_attach = registry.clone();
    let registry_cancel = registry.clone();
    let registry_data = registry;
    let sender_registry_cancel = sender_registry;

    // Get the Nova instance ID for endpoint construction
    let instance_id = nova.instance_id();

    // Handler for attach requests (typed RPC)
    // Sender calls this to attach to an anchor, receiver returns endpoint + config
    let attach_handler = NovaHandler::typed_unary::<AttachRequest, AttachResponse, _>(
        STREAM_ATTACH_HANDLER,
        move |ctx| {
            let req: AttachRequest = ctx.input;

            // Check if the receiver is available for attach
            if !registry_attach.is_available(req.anchor_id) {
                return Err(anyhow::anyhow!(
                    "anchor not found or cancelled: {}",
                    req.anchor_id
                ));
            }

            // Notify the receiver that a sender attached
            // Returns Some(timeout_option) on success, None on failure
            let message_timeout = match registry_attach.on_attach(req.anchor_id, req.sender_instance, req.session_id) {
                Some(timeout_option) => timeout_option,
                None => {
                    return Err(anyhow::anyhow!(
                        "attach rejected by receiver: {}",
                        req.anchor_id
                    ));
                }
            };

            // Calculate heartbeat interval (half of message timeout)
            let heartbeat_interval_ms = message_timeout.map(|d| d.as_millis() as u64 / 2);
            let message_timeout_ms = message_timeout.map(|d| d.as_millis() as u64);

            // Return the endpoint for data delivery
            Ok(AttachResponse {
                stream_endpoint: format!("nova://{}", instance_id),
                heartbeat_interval_ms,
                message_timeout_ms,
            })
        },
    )
    .spawn()
    .build();

    nova.register_handler(attach_handler)?;

    // Handler for cancel requests (typed RPC)
    // Receiver calls this to cancel a stream, notifies the sender
    let cancel_handler = NovaHandler::typed_unary::<CancelRequest, CancelAck, _>(
        STREAM_CANCEL_HANDLER,
        move |ctx| {
            let req: CancelRequest = ctx.input;

            // Try to cancel the sender (this is the sender-side cancel path)
            // When a receiver cancels, it sends this RPC to the sender instance.
            // We look up the sender's cancel_reason slot and set it.
            let sender_cancelled = sender_registry_cancel.cancel(
                req.anchor_id,
                req.session_id,
                req.reason.clone(),
            );

            // If no sender found, try to deliver to a local receiver
            // (for completeness, though the cancel RPC is typically sent to the sender)
            if !sender_cancelled {
                let frame_value =
                    serde_json::to_value(StreamFrame::<()>::Cancelled(req.reason.clone()))
                        .unwrap_or(serde_json::Value::Null);
                let _ = registry_cancel.deliver(req.anchor_id, frame_value);
            }

            Ok(CancelAck { received: true })
        },
    )
    .spawn()
    .build();

    nova.register_handler(cancel_handler)?;

    // Handler for data frames (AM fire-and-forget)
    // Sender calls this to deliver data frames to the receiver
    let data_handler = NovaHandler::am_handler_async(STREAM_DATA_HANDLER, move |ctx| {
        let registry = registry_data.clone();
        async move {
            // Parse the wire frame to extract anchor_id and frame
            let raw: RawWireFrame = serde_json::from_slice(&ctx.payload)
                .map_err(|e| anyhow::anyhow!("failed to parse wire frame: {}", e))?;

            // Deliver the frame to the receiver
            registry.deliver(raw.anchor_id, raw.frame)?;

            Ok(())
        }
    })
    .spawn()
    .build();

    nova.register_handler(data_handler)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_handle_is_anchor_id() {
        // StreamHandle is just a type alias for AnchorId
        let anchor_id = AnchorId::new(
            dynamo_identity::WorkerId::from_u64(42),
            100,
            1,
        );
        let handle: StreamHandle = anchor_id;
        assert_eq!(handle.owner_worker().as_u64(), 42);
        assert_eq!(handle.local_index(), 100);
        assert_eq!(handle.generation(), 1);
    }

    #[test]
    fn receiver_registry_new() {
        let registry = ReceiverRegistry::new();
        // Just verify it creates without error
        assert!(registry.receivers.is_empty());
    }
}
