// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Channel types for stream anchors.

use crate::error::{AnchorRegistryError, CancelError};
use crate::protocol::{AnchorId, MAX_ANCHOR_SLOTS, MAX_GENERATION, StreamFrame};

use dynamo_identity::{InstanceId, WorkerId};
use flume::r#async::RecvStream;
use futures::Stream;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::watch;
use uuid::Uuid;


/// Default buffer capacity for stream receivers.
pub const DEFAULT_CAPACITY: usize = 256;

/// Lifecycle states for the stream receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverState {
    /// Waiting for a sender to attach.
    AwaitingAttach,
    /// A sender is actively attached and streaming.
    Attached,
    /// Stream has been closed (finalized, timed out, or cancelled).
    Closed,
}

/// Information about the currently attached sender session.
#[derive(Debug, Clone)]
pub struct AttachedSession {
    /// The instance ID of the attached sender.
    pub sender_instance: InstanceId,
    /// Unique session ID for this attachment.
    pub session_id: Uuid,
    /// When this session attached.
    pub attached_at: Instant,
}

/// Stream receiver that creates an anchor and receives streamed data.
///
/// The receiver creates an anchor point and provides a handle that can be sent
/// to a remote sender. The sender attaches and streams data back.
///
/// # Timeout Behavior
///
/// The receiver can be configured with timeouts:
/// - **Attach timeout**: Maximum time to wait for a sender to attach
/// - **Message timeout**: Maximum time between messages while attached
///
/// When a timeout fires, a corresponding `StreamFrame::AttachTimeout` or
/// `StreamFrame::MessageTimeout` frame is delivered via `recv()`.
///
/// # Example
///
/// ```ignore
/// // Create via NovaStreaming (preferred)
/// let (receiver, handle) = streaming.create_stream_with::<MyData, _>(|b| {
///     b.attach_timeout(Duration::from_secs(30))
///      .message_timeout(Duration::from_secs(60))
/// });
///
/// loop {
///     match receiver.recv().await {
///         Some(StreamFrame::Data(item)) => process(item),
///         Some(StreamFrame::AttachTimeout) => break,
///         Some(StreamFrame::Finalized) => break,
///         None => break,
///         _ => continue,
///     }
/// }
/// ```
pub struct StreamReceiver<T> {
    inner: Arc<StreamReceiverInner<T>>,
}

/// Control context for a stream receiver.
///
/// This handle provides control operations (attach/detach/cancel) without
/// exposing the receive APIs. It can be cloned or downgraded to a weak handle.
#[derive(Clone)]
pub struct StreamReceiverContext<T> {
    inner: Arc<StreamReceiverInner<T>>,
}

/// Weak control context for a stream receiver.
pub struct StreamReceiverContextWeak<T> {
    inner: Weak<StreamReceiverInner<T>>,
}

struct StreamReceiverInner<T> {
    anchor_id: AnchorId,
    // Stored for debug/introspection but not read in normal operation
    _owner_instance: InstanceId,
    _stream_endpoint: String,
    _capacity: usize,

    // Channel for frames (data, control, timeout events)
    receiver: flume::Receiver<StreamFrame<T>>,
    sender: flume::Sender<StreamFrame<T>>,

    // State tracking
    state: Mutex<ReceiverState>,
    current_session: Mutex<Option<AttachedSession>>,
    finalized: AtomicBool,

    // Timeout configuration
    attach_timeout: Option<Duration>,
    message_timeout: Option<Duration>,

    // Deadline tracking - when the current timeout expires
    // Updated by on_attach/on_detach/on_message_received
    deadline: Mutex<Option<Instant>>,

    // State change notification for timeout task (used to trigger deadline recalc)
    state_notify: watch::Sender<u64>,

    // Anchor registry for slot management (stored for Drop)
    anchor_registry: Arc<AnchorRegistry>,

    // Optional drop hook (used to unregister from registries).
    drop_hook: OnceLock<Arc<dyn Fn(AnchorId) + Send + Sync>>,
}

/// Builder for creating StreamReceiver instances with custom configuration.
pub struct StreamReceiverBuilder {
    instance_id: InstanceId,
    stream_endpoint: String,
    anchor_registry: Arc<AnchorRegistry>,
    capacity: usize,
    attach_timeout: Option<Duration>,
    message_timeout: Option<Duration>,
}

impl StreamReceiverBuilder {
    /// Create a new builder.
    pub fn new(
        instance_id: InstanceId,
        stream_endpoint: String,
        anchor_registry: Arc<AnchorRegistry>,
    ) -> Self {
        Self {
            instance_id,
            stream_endpoint,
            anchor_registry,
            capacity: DEFAULT_CAPACITY,
            attach_timeout: None,
            message_timeout: None,
        }
    }

    /// Set the buffer capacity.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the attach timeout.
    ///
    /// If no sender attaches within this duration, an `AttachTimeout` frame
    /// is delivered. After a sender detaches, this timeout restarts.
    pub fn attach_timeout(mut self, timeout: Duration) -> Self {
        self.attach_timeout = Some(timeout);
        self
    }

    /// Set the message timeout.
    ///
    /// If no message is received within this duration while attached,
    /// a `MessageTimeout` frame is delivered. Any message (including heartbeat)
    /// resets this timer.
    pub fn message_timeout(mut self, timeout: Duration) -> Self {
        self.message_timeout = Some(timeout);
        self
    }

    /// Build the StreamReceiver.
    pub fn build<T: DeserializeOwned + Send + 'static>(self) -> StreamReceiver<T> {
        StreamReceiver::from_builder(self)
    }
}

impl<T> StreamReceiver<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create a stream adapter that yields frames from this receiver.
    pub fn into_stream(self) -> StreamReceiverStream<T> {
        let inner: RecvStream<'static, StreamFrame<T>> = self.inner.receiver.clone().into_stream();
        StreamReceiverStream {
            receiver: self,
            inner,
        }
    }

    /// Create a stream adapter that borrows this receiver.
    pub fn as_stream(&self) -> StreamReceiverRefStream<'_, T> {
        StreamReceiverRefStream {
            receiver: self,
            inner: self.inner.receiver.stream(),
        }
    }

    /// Create from builder.
    fn from_builder(builder: StreamReceiverBuilder) -> Self {
        let anchor_id = builder.anchor_registry.allocate().unwrap_or_else(|err| {
            panic!(
                "failed to allocate anchor slot for worker {}: {}",
                builder.instance_id.worker_id(),
                err
            )
        });
        let (sender, receiver) = flume::bounded(builder.capacity);
        let (state_notify, _) = watch::channel(0u64);

        // Calculate initial deadline based on attach_timeout
        let initial_deadline = builder.attach_timeout.map(|d| Instant::now() + d);

        let inner = Arc::new(StreamReceiverInner {
            anchor_id,
            _owner_instance: builder.instance_id,
            _stream_endpoint: builder.stream_endpoint,
            _capacity: builder.capacity,
            receiver,
            sender,
            state: Mutex::new(ReceiverState::AwaitingAttach),
            current_session: Mutex::new(None),
            finalized: AtomicBool::new(false),
            attach_timeout: builder.attach_timeout,
            message_timeout: builder.message_timeout,
            deadline: Mutex::new(initial_deadline),
            state_notify,
            anchor_registry: builder.anchor_registry,
            drop_hook: OnceLock::new(),
        });

        // Spawn timeout task if any timeouts are configured
        if builder.attach_timeout.is_some() || builder.message_timeout.is_some() {
            let inner_weak = Arc::downgrade(&inner);
            tokio::spawn(timeout_task(inner_weak));
        }

        Self { inner }
    }

    /// Get a control context for this receiver.
    pub fn context(&self) -> StreamReceiverContext<T> {
        StreamReceiverContext {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Get the anchor ID.
    pub fn anchor_id(&self) -> AnchorId {
        self.inner.anchor_id
    }

    /// Get the current receiver state.
    pub fn state(&self) -> ReceiverState {
        *self.inner.state.lock()
    }

    /// Get information about the currently attached session, if any.
    pub fn current_session(&self) -> Option<AttachedSession> {
        self.inner.current_session.lock().clone()
    }

    /// Get the configured message timeout (used by sender for heartbeat interval).
    pub fn message_timeout(&self) -> Option<Duration> {
        self.inner.message_timeout
    }

    /// Receive the next frame from the stream.
    ///
    /// Returns frames including:
    /// - `Data(T)` / `RawData` - Data from sender
    /// - `Attached { .. }` - Sender attached notification
    /// - `Detached` - Sender detached
    /// - `Heartbeat` - Keep-alive from sender
    /// - `Finalized` - Stream completed normally
    /// - `AttachTimeout` - No sender attached within deadline
    /// - `MessageTimeout` - No message received within deadline
    /// - `Cancelled(reason)` - Stream was cancelled
    ///
    /// Returns `None` when the stream is closed and all frames have been drained.
    pub async fn recv(&self) -> Option<StreamFrame<T>> {
        // Try to receive - even if finalized, drain remaining frames first
        match self.inner.receiver.recv_async().await {
            Ok(frame) => {
                apply_frame(self, &frame);
                Some(frame)
            }
            Err(_) => None, // Channel closed (all frames drained)
        }
    }

    /// Try to receive without blocking.
    ///
    /// Returns `None` if no frame is available or channel is closed.
    pub fn try_recv(&self) -> Option<StreamFrame<T>> {
        // Try to receive - even if finalized, drain remaining frames first
        self.inner.receiver.try_recv().ok().inspect(|frame| {
            apply_frame(self, frame);
        })
    }

    /// Check if the stream has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.inner.finalized.load(Ordering::SeqCst)
    }

    /// Cancel the stream with a reason.
    ///
    /// This method:
    /// 1. Delivers a `Cancelled` frame to `recv()`
    /// 2. Marks the stream as closed
    /// 3. Returns the cancelled session info (if attached) so the caller
    ///    can notify the sender via transport
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Cancel and notify sender via transport
    /// if let Some(session) = receiver.cancel("user requested abort")? {
    ///     transport.send_cancel(
    ///         session.sender_instance,
    ///         receiver.anchor_id(),
    ///         session.session_id,
    ///         "user requested abort".to_string(),
    ///     ).await;
    /// }
    /// ```
    ///
    /// # Returns
    /// - `Ok(Some(session))` if a sender was attached (caller should notify)
    /// - `Ok(None)` if no sender was attached
    /// - `Err(CancelError::AlreadyClosed)` if stream was already closed
    pub fn cancel(&self, reason: impl Into<String>) -> Result<Option<AttachedSession>, CancelError> {
        self.inner.cancel(reason.into())
    }

    /// Get the internal sender for use by transport layer.
    ///
    /// Transport implementations should use this to deliver frames to the receiver.
    /// When delivering frames, also call `on_message_received()` to reset timeouts.
    pub fn frame_sender(&self) -> flume::Sender<StreamFrame<T>> {
        self.inner.sender.clone()
    }

    /// Called by transport layer when a sender attaches.
    ///
    /// This should be called when an Attach control message is received.
    /// It transitions the receiver to `Attached` state and delivers an
    /// `Attached` frame to `recv()`.
    pub fn on_attach(&self, sender_instance: InstanceId, session_id: Uuid) {
        let _ = self.inner.try_on_attach(sender_instance, session_id);
    }

    /// Called by transport layer when a sender detaches.
    ///
    /// Transitions receiver back to `AwaitingAttach` state and delivers
    /// a `Detached` frame. The attach timeout will restart.
    pub fn on_detach(&self, session_id: Uuid) {
        self.inner.on_detach(session_id);
    }

    /// Called by transport layer when a message is received.
    ///
    /// This resets the message timeout timer. Should be called for any
    /// frame received from the sender (including heartbeats).
    pub fn on_message_received(&self) {
        self.inner.on_message_received();
    }

    /// Close the stream with a terminal frame.
    ///
    /// Used by transport layer to close the stream with a specific frame
    /// (e.g., `Finalized`, `Error`, etc.).
    pub fn close_with_frame(&self, frame: StreamFrame<T>) {
        self.inner.close_with_frame(frame);
    }

    pub(crate) fn set_drop_hook(&self, hook: Arc<dyn Fn(AnchorId) + Send + Sync>) -> bool {
        self.inner.drop_hook.set(hook).is_ok()
    }
}

impl<T> StreamReceiverContext<T> {
    /// Downgrade this context to a weak handle.
    pub fn downgrade(&self) -> StreamReceiverContextWeak<T> {
        StreamReceiverContextWeak {
            inner: Arc::downgrade(&self.inner),
        }
    }
}

impl<T> Clone for StreamReceiverContextWeak<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> StreamReceiverContextWeak<T> {
    /// Upgrade this weak handle to a strong context if the receiver is alive.
    pub fn upgrade(&self) -> Option<StreamReceiverContext<T>> {
        self.inner
            .upgrade()
            .map(|inner| StreamReceiverContext { inner })
    }
}

impl<T> StreamReceiverContext<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Get the receiver's frame sender for transport delivery.
    pub fn frame_sender(&self) -> flume::Sender<StreamFrame<T>> {
        self.inner.sender.clone()
    }

    /// Get the receiver's configured message timeout.
    pub fn message_timeout(&self) -> Option<Duration> {
        self.inner.message_timeout
    }

    /// Called by transport layer when a sender attaches.
    pub fn on_attach(&self, sender_instance: InstanceId, session_id: Uuid) -> bool {
        self.inner.try_on_attach(sender_instance, session_id)
    }

    /// Called by transport layer when a sender detaches.
    pub fn on_detach(&self, session_id: Uuid) -> bool {
        self.inner.on_detach(session_id);
        true
    }

    /// Called by transport layer when a message is received.
    pub fn on_message_received(&self) -> bool {
        self.inner.on_message_received();
        true
    }

    /// Cancel the stream with a reason.
    pub fn cancel(&self, reason: impl Into<String>) -> Result<Option<AttachedSession>, CancelError> {
        self.inner.cancel(reason.into())
    }
}

fn apply_frame<T>(receiver: &StreamReceiver<T>, frame: &StreamFrame<T>)
where
    T: DeserializeOwned + Send + 'static,
{
    match frame {
        StreamFrame::Detached => receiver.inner.apply_detach(),
        StreamFrame::Data(_) | StreamFrame::RawData(_) | StreamFrame::Heartbeat => {
            receiver.inner.on_message_received();
        }
        _ => {}
    }

    if frame.is_terminal() {
        receiver.inner.finalized.store(true, Ordering::SeqCst);
        *receiver.inner.state.lock() = ReceiverState::Closed;
    }
}

/// Stream adapter that owns the receiver.
pub struct StreamReceiverStream<T: 'static> {
    receiver: StreamReceiver<T>,
    inner: RecvStream<'static, StreamFrame<T>>,
}

/// Stream adapter that borrows the receiver.
pub struct StreamReceiverRefStream<'a, T> {
    receiver: &'a StreamReceiver<T>,
    inner: RecvStream<'a, StreamFrame<T>>,
}

impl<T> Stream for StreamReceiverStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    type Item = StreamFrame<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(frame)) => {
                apply_frame(&self.receiver, &frame);
                Poll::Ready(Some(frame))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<'a, T> Stream for StreamReceiverRefStream<'a, T>
where
    T: DeserializeOwned + Send + 'static,
{
    type Item = StreamFrame<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(frame)) => {
                apply_frame(self.receiver, &frame);
                Poll::Ready(Some(frame))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> Drop for StreamReceiverInner<T> {
    fn drop(&mut self) {
        if let Some(hook) = self.drop_hook.get() {
            hook(self.anchor_id);
        }
        self.anchor_registry.release(self.anchor_id);
    }
}

impl<T> StreamReceiverInner<T>
where
    T: DeserializeOwned + Send + 'static,
{
    fn try_on_attach(&self, sender_instance: InstanceId, session_id: Uuid) -> bool {
        let mut state = self.state.lock();
        if *state != ReceiverState::AwaitingAttach {
            return false;
        }

        let session = AttachedSession {
            sender_instance,
            session_id,
            attached_at: Instant::now(),
        };

        *self.current_session.lock() = Some(session.clone());
        *state = ReceiverState::Attached;

        // Set new deadline for message timeout
        *self.deadline.lock() = self.message_timeout.map(|d| Instant::now() + d);

        // Notify timeout task of state change
        self.state_notify.send_modify(|v| *v = v.wrapping_add(1));

        // Deliver Attached frame
        let _ = self.sender.try_send(StreamFrame::Attached {
            sender_instance,
            session_id,
        });

        true
    }

    fn on_detach(&self, _session_id: Uuid) {
        self.apply_detach();
        let _ = self.sender.try_send(StreamFrame::Detached);
    }

    fn apply_detach(&self) {
        let mut state = self.state.lock();
        if *state == ReceiverState::Closed {
            return;
        }

        *self.current_session.lock() = None;
        *state = ReceiverState::AwaitingAttach;

        // Reset deadline for attach timeout
        *self.deadline.lock() = self.attach_timeout.map(|d| Instant::now() + d);

        // Notify timeout task of state change
        self.state_notify.send_modify(|v| *v = v.wrapping_add(1));
    }

    fn on_message_received(&self) {
        // Reset deadline for message timeout (only if attached)
        if *self.state.lock() == ReceiverState::Attached
            && let Some(timeout) = self.message_timeout
        {
            *self.deadline.lock() = Some(Instant::now() + timeout);
        }
    }

    fn cancel(&self, reason: String) -> Result<Option<AttachedSession>, CancelError> {
        // Check if already closed
        if self.finalized.load(Ordering::SeqCst) {
            return Err(CancelError::AlreadyClosed);
        }

        // Deliver Cancelled frame to recv()
        let _ = self
            .sender
            .try_send(StreamFrame::Cancelled(reason.clone()));

        // Mark as closed
        self.finalized.store(true, Ordering::SeqCst);
        *self.state.lock() = ReceiverState::Closed;

        // Take the session info (if attached) for caller to notify sender
        let session = self.current_session.lock().take();

        // Notify timeout task to stop
        self.state_notify.send_modify(|v| *v = v.wrapping_add(1));

        Ok(session)
    }

    fn close_with_frame(&self, frame: StreamFrame<T>) {
        self.finalized.store(true, Ordering::SeqCst);
        *self.state.lock() = ReceiverState::Closed;
        let _ = self.sender.try_send(frame);
    }
}

/// Background task that handles timeout delivery.
///
/// This task monitors the receiver's deadline and delivers timeout frames when
/// the configured timeouts expire without activity.
async fn timeout_task<T: Send + 'static>(inner: std::sync::Weak<StreamReceiverInner<T>>) {
    // Determine check interval (minimum of configured timeouts / 2, at least 10ms)
    let check_interval = {
        let Some(inner_arc) = inner.upgrade() else {
            return;
        };
        let attach = inner_arc.attach_timeout.unwrap_or(Duration::from_secs(60));
        let message = inner_arc.message_timeout.unwrap_or(Duration::from_secs(60));
        let min_timeout = attach.min(message);
        // Check at half the timeout interval, at least every 10ms
        (min_timeout / 2).max(Duration::from_millis(10))
    };

    loop {
        tokio::time::sleep(check_interval).await;

        let Some(inner_arc) = inner.upgrade() else {
            return; // Receiver dropped
        };

        let current_state = *inner_arc.state.lock();

        // Check if closed
        if current_state == ReceiverState::Closed {
            return;
        }

        // Check if deadline passed
        if let Some(deadline) = *inner_arc.deadline.lock()
            && Instant::now() >= deadline
        {
            // Timeout fired
            let frame = match current_state {
                ReceiverState::AwaitingAttach => StreamFrame::AttachTimeout,
                ReceiverState::Attached => StreamFrame::MessageTimeout,
                ReceiverState::Closed => return,
            };

            inner_arc.finalized.store(true, Ordering::SeqCst);
            *inner_arc.state.lock() = ReceiverState::Closed;
            let _ = inner_arc.sender.try_send(frame);
            return;
        }
    }
}

/// Registry for managing anchor slots.
///
/// This follows the same slot arena pattern as nova's ResponseManager,
/// with generational IDs to prevent ABA problems.
pub struct AnchorRegistry {
    worker_id: WorkerId,
    slots: Vec<AnchorSlot>,
    free_list: Mutex<VecDeque<u16>>,
}

struct AnchorSlot {
    generation: AtomicU64,
    active: std::sync::atomic::AtomicBool,
}

impl AnchorRegistry {
    /// Create a new anchor registry.
    pub fn new(worker_id: WorkerId) -> Self {
        let slots: Vec<AnchorSlot> = (0..MAX_ANCHOR_SLOTS)
            .map(|_| AnchorSlot {
                generation: AtomicU64::new(0),
                active: std::sync::atomic::AtomicBool::new(false),
            })
            .collect();

        let free_list: VecDeque<u16> = (0..MAX_ANCHOR_SLOTS as u16).collect();

        Self {
            worker_id,
            slots,
            free_list: Mutex::new(free_list),
        }
    }

    /// Allocate a new anchor ID.
    pub fn allocate(&self) -> Result<AnchorId, AnchorRegistryError> {
        let mut free_list = self.free_list.lock();

        let index = free_list
            .pop_front()
            .ok_or(AnchorRegistryError::Exhausted)?;

        let slot = &self.slots[index as usize];
        let generation = slot.generation.load(Ordering::SeqCst);

        if generation > MAX_GENERATION {
            // Slot exhausted, don't return to free list
            return Err(AnchorRegistryError::GenerationExhausted);
        }

        slot.active.store(true, Ordering::SeqCst);

        Ok(AnchorId::new(self.worker_id, index, generation))
    }

    /// Release an anchor ID back to the pool.
    pub fn release(&self, anchor_id: AnchorId) {
        let index = anchor_id.local_index();
        if (index as usize) >= self.slots.len() {
            return;
        }

        let slot = &self.slots[index as usize];
        let expected_gen = anchor_id.generation();
        let current_gen = slot.generation.load(Ordering::SeqCst);

        // Only release if generation matches
        if current_gen != expected_gen {
            return;
        }

        // Increment generation
        let new_gen = current_gen + 1;
        slot.generation.store(new_gen, Ordering::SeqCst);
        slot.active.store(false, Ordering::SeqCst);

        // Return to free list if not exhausted
        if new_gen <= MAX_GENERATION {
            let mut free_list = self.free_list.lock();
            free_list.push_back(index);
        }
    }

    /// Check if an anchor ID is currently valid.
    #[allow(dead_code)]
    pub fn is_valid(&self, anchor_id: AnchorId) -> bool {
        let index = anchor_id.local_index();
        if (index as usize) >= self.slots.len() {
            return false;
        }

        let slot = &self.slots[index as usize];
        let current_gen = slot.generation.load(Ordering::SeqCst);

        anchor_id.generation() == current_gen && slot.active.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_registry_allocate_release() {
        let registry = AnchorRegistry::new(WorkerId::from_u64(42));

        let id1 = registry.allocate().unwrap();
        assert!(registry.is_valid(id1));

        registry.release(id1);
        assert!(!registry.is_valid(id1));

        // After release, generation is incremented so old ID is invalid
        // The released slot goes to the back of the queue (FIFO),
        // so the next allocation may return a different slot
        let id2 = registry.allocate().unwrap();
        assert!(registry.is_valid(id2));
    }

    #[test]
    fn anchor_registry_stale_id_rejected() {
        let registry = AnchorRegistry::new(WorkerId::from_u64(42));

        let id1 = registry.allocate().unwrap();
        registry.release(id1);

        // Old ID should no longer be valid
        assert!(!registry.is_valid(id1));

        // New allocation should work
        let id2 = registry.allocate().unwrap();
        assert!(registry.is_valid(id2));
    }
}
