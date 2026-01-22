// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use dashmap::DashMap;
use dynamo_identity::{InstanceId, WorkerId};
use dynamo_nova_discovery::peer::{DiscoveryQueryError, PeerDiscoveryManager};
use futures::future::Either;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio_util::task::TaskTracker;
use tracing::warn;

use crate::am::common::events::{EventType, Outcome, encode_event_header};
use crate::am::common::responses::{ResponseAwaiter, ResponseId, ResponseManager};
use crate::events::{
    CompletionKind, EventHandle, EventPoison, EventStatus, LocalEvent, LocalEventSystem,
};
use dynamo_nova_backend::MessageType;

/// Minimal sending surface needed by the event system.
pub(crate) trait EventMessenger: Send + Sync {
    fn send_system(&self, target: InstanceId, handler: &str, payload: Bytes) -> Result<()>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventSubscribeMessage {
    pub handle: u128,
    pub subscriber_worker: u64,
    pub subscriber_instance: InstanceId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventCompletionMessage {
    pub handle: u128,
    pub poisoned: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EventTriggerRequestMessage {
    pub handle: u128,
    pub requester_worker: u64,
    pub requester_instance: InstanceId,
    pub poisoned: Option<String>,
    pub response_id: Option<u128>, // New: for ResponseManager completion
}

/// Cached information for completed remote events stored in the LRU cache.
///
/// Purpose: Enables fast-path lookups for recently completed remote events
/// without needing to query the remote node or maintain active RemoteEvent entries.
///
/// # Memory Management
///
/// Minimal memory footprint:
/// - `highest_generation`: Tracks the highest completed generation
/// - `poisoned_generations`: Index-only set of ALL poisoned generations (no error strings)
///
/// # Fast-Path Logic
///
/// For a requested generation:
/// - If `generation > highest_generation` → not complete, fallback to network
/// - If `generation < (highest_generation - 10)` → too old, fallback to network for accuracy
/// - If within last 10 generations:
///   - If in `poisoned_generations` → return Poisoned (generic error message)
///   - Else → return Triggered
struct CompletedEventInfo {
    /// Highest generation known to be complete for this event
    highest_generation: u32,
    /// Index of ALL poisoned generations (without error messages to save memory)
    poisoned_generations: std::collections::BTreeSet<u32>,
}

/// Per-remote-handle state on the subscriber side.
///
/// A RemoteEvent represents a remote event that this node is subscribing to.
/// It maintains state for pending subscriptions, active waiters, and completion history.
///
/// # Lifecycle and Memory Management
///
/// RemoteEvents are stored in the `remote_events` DashMap while they have active waiters
/// or pending subscriptions. Once all pending generations are complete and all waiters
/// have been notified, the RemoteEvent is moved to the LRU cache and removed from the
/// active DashMap to prevent unbounded memory growth.
///
/// # Fields
///
/// * `local_system` - Creates LocalEvents for each remote generation being waited on.
///   This allows remote event waiters to use the same awaiter API as local events.
///
/// * `known_triggered` - Fast-path optimization tracking the highest generation known
///   to be complete. If a waiter requests generation <= this value, we can immediately
///   return from cache without spawning tasks or sending network messages.
///
/// * `local_events` - Active waiters for specific generations. Maps generation -> LocalEvent.
///   **Memory**: Entries are added when waiters arrive and removed in `complete_generation()`
///   after triggering/poisoning. Cleaned up when moved to LRU cache.
///
/// * `completions` - History cache of completion results (triggered vs poisoned).
///   Maps generation -> CompletionKind. Allows returning correct completion status
///   for generations that have already completed.
///   **Memory**: Grows with each completed generation. Cleaned up when moved to LRU cache.
///
/// * `pending` - Deduplication set preventing multiple subscription requests for the
///   same generation. We only send one `_event_subscribe` message per generation.
///   **Memory**: Grows with each new generation subscribed. Cleaned up when moved to LRU cache.
struct RemoteEvent {
    local_system: Arc<LocalEventSystem>,
    known_triggered: AtomicU32,
    local_events: Mutex<BTreeMap<u32, LocalEvent>>,
    completions: Mutex<BTreeMap<u32, Arc<CompletionKind>>>,
    pending: Mutex<BTreeSet<u32>>,
}

impl RemoteEvent {
    fn new(local_system: Arc<LocalEventSystem>) -> Self {
        assert_ne!(
            local_system.worker_id(),
            0,
            "Local event system must be a local-only system"
        );
        Self {
            local_system,
            known_triggered: AtomicU32::new(0),
            local_events: Mutex::new(BTreeMap::new()),
            completions: Mutex::new(BTreeMap::new()),
            pending: Mutex::new(BTreeSet::new()),
        }
    }

    fn known_generation(&self) -> u32 {
        self.known_triggered.load(Ordering::Acquire)
    }

    fn update_known_generation(&self, generation: u32) {
        let _ = self
            .known_triggered
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (generation > current).then_some(generation)
            });
    }

    fn register_waiter(&self, generation: u32) -> WaitRegistration {
        if generation <= self.known_generation() {
            let completions = self.completions.lock();
            return match completions.get(&generation) {
                Some(completion) => match completion.as_ref() {
                    CompletionKind::Triggered => WaitRegistration::Ready,
                    CompletionKind::Poisoned(p) => WaitRegistration::Poisoned(p.clone()),
                },
                None => WaitRegistration::Ready, // Triggered without poison
            };
        }

        // Create or get LocalEvent for this generation
        let local_event = {
            let mut local_events = self.local_events.lock();
            local_events
                .entry(generation)
                .or_insert_with(|| {
                    // Create a new LocalEvent for this remote generation
                    self.local_system
                        .new_event()
                        .expect("Failed to create local event for remote subscription")
                })
                .clone()
        };

        // Return the LocalEvent's waiter
        let waiter = self
            .local_system
            .awaiter(local_event.handle())
            .expect("Failed to create waiter for local event");
        WaitRegistration::Pending(waiter)
    }

    fn add_pending(&self, generation: u32) -> bool {
        self.pending.lock().insert(generation)
    }

    fn complete_generation(&self, generation: u32, completion: CompletionKind) {
        self.update_known_generation(generation);

        // Store completion with bounded history to prevent unbounded growth
        {
            let mut completions = self.completions.lock();
            completions.insert(generation, Arc::new(completion.clone()));

            // Prune to last 100 generations to prevent memory leak on long-lived RemoteEvents
            const MAX_COMPLETION_HISTORY: u32 = 100;
            let known_gen = self.known_generation();
            if known_gen > MAX_COMPLETION_HISTORY {
                completions.retain(|&g, _| g > known_gen - MAX_COMPLETION_HISTORY);
            }
        }

        // Trigger or poison all LocalEvents for generations <= this one
        let events_to_complete = {
            let mut local_events = self.local_events.lock();
            let gens_to_wake: Vec<u32> =
                local_events.range(..=generation).map(|(g, _)| *g).collect();

            let mut to_complete = Vec::new();
            for generation_to_wake in gens_to_wake {
                if let Some(event) = local_events.remove(&generation_to_wake) {
                    to_complete.push(event);
                }
            }
            to_complete
        };

        // Complete LocalEvents outside lock
        for event in events_to_complete {
            match &completion {
                CompletionKind::Triggered => {
                    let _ = event.trigger();
                }
                CompletionKind::Poisoned(p) => {
                    let _ = self.local_system.poison(event.handle(), p.reason());
                }
            }
        }

        // Clean up pending
        let mut pending = self.pending.lock();
        let to_remove: Vec<u32> = pending
            .iter()
            .copied()
            .take_while(|g| *g <= generation)
            .collect();
        for g in to_remove {
            pending.remove(&g);
        }
    }

    fn status_for(&self, generation: u32) -> EventStatus {
        if generation <= self.known_generation() {
            let completions = self.completions.lock();
            match completions.get(&generation) {
                Some(completion) => match completion.as_ref() {
                    CompletionKind::Triggered => EventStatus::Ready,
                    CompletionKind::Poisoned(_) => EventStatus::Poisoned,
                },
                None => EventStatus::Ready, // Triggered without poison
            }
        } else {
            EventStatus::Pending
        }
    }
}

enum WaitRegistration {
    Ready,
    Pending(crate::events::LocalEventWaiter),
    Poisoned(Arc<EventPoison>),
}

/// Nova event system that extends local events with remote capabilities.
///
/// NovaEvents provides distributed event capabilities on top of the local event system,
/// allowing events owned by one Nova instance to be waited on, triggered, or poisoned
/// from other Nova instances across the network.
///
/// # Architecture
///
/// The system uses three main data structures for memory management:
///
/// ## 1. `remote_events` - Active Remote Subscriptions
///
/// DashMap tracking RemoteEvent state for events this node is currently subscribing to.
///
/// **When entries are added**: When this node calls `wait_remote()`, `trigger_remote()`,
/// or `poison_remote()` for an event owned by another node.
///
/// **When entries are removed**: When all pending generations are complete and all active
/// waiters have been notified. The RemoteEvent is then moved to the LRU cache.
///
/// **Memory growth**: Bounded by the number of distinct remote events currently being
/// subscribed to. Entries are automatically cleaned up via `maybe_cache_remote_event()`.
///
/// ## 2. `completed_cache` - LRU Cache of Completed Events
///
/// LRU cache (default 1000 entries) storing completion info for recently completed remote
/// events. Enables fast-path lookups without network queries or active RemoteEvent state.
///
/// **When entries are added**: When a RemoteEvent has no more active waiters or pending
/// subscriptions and is migrated from `remote_events`.
///
/// **When entries are removed**: Automatically evicted by LRU policy when cache is full.
///
/// **Memory growth**: Bounded by cache size (1000 entries). Each entry is small
/// (EventKey + highest_generation + completion history).
///
/// ## 3. `owner_subscribers` - Subscription Task Registry
///
/// DashMap tracking spawned notification tasks for events owned by THIS node.
/// Maps EventKey -> (InstanceId -> highest_generation_subscribed).
///
/// **When entries are added**: When a remote node subscribes to an event owned by this
/// node via `handle_subscribe()`. A task is spawned to wait on the local event and send
/// a completion notification to the subscriber.
///
/// **When entries are removed**: After the spawned task successfully sends a completion
/// notification to the remote subscriber via `cleanup_owner_subscription()`.
///
/// **Memory growth**: Bounded by the number of active remote subscriptions to locally-owned
/// events. Each entry tracks one spawned notification task per subscribing instance.
///
/// **Deduplication**: Prevents spawning multiple tasks for the same (event, subscriber)
/// pair. If subscriber requests generation N, and we already have a task for generation M >= N,
/// the existing task will handle it.
///
/// # 3-Tier Lookup Pattern
///
/// For remote operations (wait/trigger/poison), the system uses a 3-tier lookup:
/// 1. **TIER 1**: Check `completed_cache` (fast path, no allocation/network)
/// 2. **TIER 2**: Check `remote_events` (active subscription exists)
/// 3. **TIER 3**: Create subscription and query remote node (slow path)
pub struct NovaEvents {
    local: Arc<LocalEventSystem>,
    worker_id: WorkerId,
    instance_id: InstanceId,
    backend: Arc<dynamo_nova_backend::NovaBackend>,
    messenger: RwLock<Option<Arc<dyn EventMessenger>>>,
    discovery: Arc<PeerDiscoveryManager>,
    remote_events: DashMap<crate::events::EventKey, Arc<RemoteEvent>>,
    completed_cache: Arc<Mutex<LruCache<crate::events::EventKey, CompletedEventInfo>>>,
    owner_subscribers: DashMap<crate::events::EventKey, DashMap<InstanceId, u32>>,
    response_manager: Arc<ResponseManager>,
    tasks: TaskTracker,
}

impl NovaEvents {
    pub(crate) fn new(
        instance_id: InstanceId,
        local: Arc<LocalEventSystem>,
        backend: Arc<dynamo_nova_backend::NovaBackend>,
        response_manager: Arc<ResponseManager>,
        discovery: Arc<PeerDiscoveryManager>,
    ) -> Arc<Self> {
        const DEFAULT_COMPLETED_CACHE_SIZE: usize = 1000;

        let worker_id = instance_id.worker_id();
        assert_eq!(
            worker_id.as_u64(),
            local.worker_id(),
            "Worker ID must match local event system: {} != {}",
            worker_id.as_u64(),
            local.worker_id()
        );

        assert_ne!(worker_id.as_u64(), 0, "Worker ID must be non-zero");

        Arc::new(Self {
            local,
            worker_id,
            instance_id,
            backend,
            messenger: RwLock::new(None),
            discovery,
            remote_events: DashMap::new(),
            completed_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPLETED_CACHE_SIZE).unwrap(),
            ))),
            owner_subscribers: DashMap::new(),
            response_manager,
            tasks: TaskTracker::new(),
        })
    }

    #[allow(dead_code)]
    pub(crate) fn worker_id(&self) -> WorkerId {
        self.worker_id
    }

    pub fn local(&self) -> &Arc<LocalEventSystem> {
        &self.local
    }

    #[allow(dead_code)]
    pub(crate) fn task_tracker(&self) -> &TaskTracker {
        &self.tasks
    }

    pub(crate) fn set_messenger(&self, messenger: Arc<dyn EventMessenger>) {
        *self.messenger.write() = Some(messenger);
    }

    /// Resolve instance_id from worker_id using backend cache + discovery fallback.
    ///
    /// Fast path: Check backend's worker cache (populated via register_peer)
    /// Slow path: Query discovery system and auto-register peer with backend
    async fn resolve_instance_id(&self, worker_id: WorkerId) -> Result<InstanceId> {
        // Fast path: backend worker cache for registered peers
        if let Ok(instance_id) = self.backend.try_translate_worker_id(worker_id) {
            return Ok(instance_id);
        }

        // Slow path: discovery query (may be async remote lookup)
        let either_result = self.discovery.discover_by_worker_id(worker_id).await;
        let query_result = match either_result {
            Either::Left(ready) => ready.into_inner(),
            Either::Right(shared_future) => shared_future.await,
        };

        match query_result {
            Ok(peer_info) => {
                let instance_id = peer_info.instance_id();
                // Auto-register discovered peer with backend for future fast-path lookups
                self.backend.register_peer(peer_info)?;
                Ok(instance_id)
            }
            Err(DiscoveryQueryError::NotFound) => {
                bail!("No discovery entry for worker {}", worker_id)
            }
            Err(DiscoveryQueryError::Backend(e)) => {
                bail!("Discovery backend error for worker {}: {}", worker_id, e)
            }
        }
    }

    pub fn new_event(self: &Arc<Self>) -> Result<LocalEvent> {
        self.local.new_event()
    }

    pub fn create_user_event(self: &Arc<Self>) -> Result<LocalEvent> {
        self.local.create_user_event()
    }

    pub fn awaiter(
        self: &Arc<Self>,
        handle: EventHandle,
    ) -> Result<crate::events::LocalEventWaiter> {
        let owner_worker = handle.owner_worker();
        assert_ne!(
            owner_worker.as_u64(),
            0,
            "Invalid event; local-only events can not be used as nova events"
        );
        if owner_worker == self.worker_id {
            self.local.awaiter(handle)
        } else {
            self.wait_remote(handle)
        }
    }

    pub fn poll(self: &Arc<Self>, handle: EventHandle) -> Result<EventStatus> {
        let owner_worker = handle.owner_worker();
        assert_ne!(
            owner_worker.as_u64(),
            0,
            "Invalid event; local-only events can not be used as nova events"
        );
        if owner_worker == self.worker_id {
            self.local.poll(handle)
        } else {
            self.poll_remote(handle)
        }
    }

    pub async fn trigger(self: &Arc<Self>, handle: EventHandle) -> Result<()> {
        let owner_worker = handle.owner_worker();
        assert_ne!(
            owner_worker.as_u64(),
            0,
            "Invalid event; local-only events can not be used as nova events"
        );
        if owner_worker == self.worker_id {
            self.local.trigger(handle)
        } else {
            let mut awaiter = self.trigger_remote(handle)?;
            awaiter.recv().await.map_err(|e| anyhow!("{}", e))?;
            Ok(())
        }
    }

    pub async fn poison(
        self: &Arc<Self>,
        handle: EventHandle,
        reason: impl Into<String>,
    ) -> Result<()> {
        let owner_worker = handle.owner_worker();
        assert_ne!(
            owner_worker.as_u64(),
            0,
            "Invalid event; local-only events can not be used as nova events"
        );
        if owner_worker == self.worker_id {
            self.local.poison(handle, reason)
        } else {
            let mut awaiter = self.poison_remote(handle, reason.into())?;
            awaiter.recv().await.map_err(|e| anyhow!("{}", e))?;
            Ok(())
        }
    }

    pub fn merge_events(self: &Arc<Self>, inputs: Vec<EventHandle>) -> Result<EventHandle> {
        for handle in &inputs {
            let owner_worker = handle.owner_worker();
            assert_ne!(
                owner_worker.as_u64(),
                0,
                "Invalid event; local-only events can not be used as nova events"
            );
            if owner_worker != self.worker_id {
                bail!("Merge only supports local events; got {}", handle);
            }
        }
        self.local.merge_events(inputs)
    }

    // Owner-side handlers
    pub(crate) fn handle_subscribe(self: &Arc<Self>, payload: Bytes) -> Result<()> {
        let message: EventSubscribeMessage = serde_json::from_slice(&payload)?;
        let handle = EventHandle::from_raw(message.handle);
        let owner_worker = handle.owner_worker();
        assert_ne!(
            owner_worker.as_u64(),
            0,
            "Invalid event; local-only events can not be used as nova events"
        );
        if owner_worker != self.worker_id {
            bail!("Subscribe for non-local event {}", handle);
        }

        match self.local.poll(handle)? {
            EventStatus::Ready => self.send_completion(
                handle,
                message.subscriber_instance,
                CompletionKind::Triggered,
            ),
            EventStatus::Poisoned => {
                let reason = self
                    .local
                    .poison_reason(handle)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown poison".to_string());
                let poison = Arc::new(EventPoison::new(handle, reason));
                self.send_completion(
                    handle,
                    message.subscriber_instance,
                    CompletionKind::Poisoned(poison),
                )
            }
            EventStatus::Pending => {
                self.register_owner_subscription(handle, message.subscriber_instance)?;
                Ok(())
            }
        }
    }

    pub(crate) fn handle_trigger(&self, payload: Bytes) -> Result<()> {
        let message: EventCompletionMessage = serde_json::from_slice(&payload)?;
        let handle = EventHandle::from_raw(message.handle);

        let completion = if let Some(reason) = message.poisoned {
            CompletionKind::Poisoned(Arc::new(EventPoison::new(handle, reason)))
        } else {
            CompletionKind::Triggered
        };

        let key = crate::events::EventKey::from_handle(handle);
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.local.clone())))
            .clone();

        // complete_generation now handles notification internally
        remote_event.complete_generation(handle.generation(), completion);

        // Check if RemoteEvent can be moved to LRU cache
        self.maybe_cache_remote_event(key, remote_event);

        Ok(())
    }

    pub(crate) fn handle_trigger_request(self: &Arc<Self>, payload: Bytes) -> Result<()> {
        let message: EventTriggerRequestMessage = serde_json::from_slice(&payload)?;
        let handle = EventHandle::from_raw(message.handle);
        if handle.owner_worker() != self.worker_id {
            bail!("Trigger request for non-local event {}", handle);
        }

        // If there's a response_id, we need to send response via network messages
        if let Some(response_id_raw) = message.response_id {
            let response_id = ResponseId::from_u128(response_id_raw);

            match self.local.poll(handle)? {
                EventStatus::Ready | EventStatus::Poisoned => {
                    // Event already complete - send completion to any existing subscribers
                    let completion = match self.local.poll(handle)? {
                        EventStatus::Poisoned => {
                            let reason = self
                                .local
                                .poison_reason(handle)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| "unknown poison".to_string());
                            CompletionKind::Poisoned(Arc::new(EventPoison::new(handle, reason)))
                        }
                        _ => CompletionKind::Triggered,
                    };
                    self.send_completion(handle, message.requester_instance, completion)?;

                    // Also send ACK/NACK via response channel
                    let system = Arc::clone(self);
                    self.tasks.spawn(async move {
                        if system.local.poll(handle).ok() == Some(EventStatus::Poisoned) {
                            let reason = system
                                .local
                                .poison_reason(handle)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| "unknown poison".to_string());
                            let _ = system.send_nack(response_id, reason).await;
                        } else {
                            let _ = system.send_ack(response_id).await;
                        }
                    });
                    Ok(())
                }
                EventStatus::Pending => {
                    // Register subscription first
                    self.register_owner_subscription(handle, message.requester_instance)?;

                    // Determine if this is a poison or trigger request
                    let is_poison_request = message.poisoned.is_some();

                    // Apply the trigger/poison operation
                    if let Some(reason) = message.poisoned {
                        self.local.poison(handle, reason)?;
                    } else {
                        self.local.trigger(handle)?;
                    }

                    // Spawn task to wait for completion and send ACK/NACK based on result
                    // Lenient semantics: poison request ACKs if event completes (any outcome)
                    let system = Arc::clone(self);
                    self.tasks.spawn(async move {
                        // Wait for the event to actually complete
                        match system.local.awaiter(handle) {
                            Ok(waiter) => match waiter.await {
                                Ok(()) => {
                                    // Event triggered successfully
                                    if is_poison_request {
                                        // Lenient: poison request succeeded (event is complete)
                                        let _ = system.send_ack(response_id).await;
                                    } else {
                                        // Trigger request succeeded - ACK
                                        let _ = system.send_ack(response_id).await;
                                    }
                                }
                                Err(poison_err) => {
                                    // Event was poisoned
                                    if is_poison_request {
                                        // Poison request succeeded (event is complete) - ACK
                                        let _ = system.send_ack(response_id).await;
                                    } else {
                                        // Trigger request but event poisoned - NACK
                                        let _ = system
                                            .send_nack(response_id, poison_err.to_string())
                                            .await;
                                    }
                                }
                            },
                            Err(e) => {
                                // Failed to create awaiter - send NACK
                                let _ = system.send_nack(response_id, e.to_string()).await;
                            }
                        }
                    });
                    Ok(())
                }
            }
        } else {
            // Legacy path without response_id (for subscribe-based waiting)
            match self.local.poll(handle)? {
                EventStatus::Ready => self.send_completion(
                    handle,
                    message.requester_instance,
                    CompletionKind::Triggered,
                ),
                EventStatus::Poisoned => {
                    let reason = self
                        .local
                        .poison_reason(handle)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown poison".to_string());
                    let poison = Arc::new(EventPoison::new(handle, reason));
                    self.send_completion(
                        handle,
                        message.requester_instance,
                        CompletionKind::Poisoned(poison),
                    )
                }
                EventStatus::Pending => {
                    self.register_owner_subscription(handle, message.requester_instance)?;
                    if let Some(reason) = message.poisoned {
                        self.local.poison(handle, reason)?;
                    } else {
                        self.local.trigger(handle)?;
                    }
                    Ok(())
                }
            }
        }
    }

    // Subscriber-side paths
    fn wait_remote(
        self: &Arc<Self>,
        handle: EventHandle,
    ) -> Result<crate::events::LocalEventWaiter> {
        let key = crate::events::EventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache first (fast path for recently completed events)
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key) {
                if generation > info.highest_generation {
                    // Not yet complete - fallback to TIER 2/3
                } else if generation < info.highest_generation.saturating_sub(10) {
                    // Too old (not in recent 10) - fallback to TIER 2/3 for accurate data
                } else {
                    // Within recent 10 generations - can answer from cache
                    if info.poisoned_generations.contains(&generation) {
                        // Poisoned (use generic error since we don't store original strings)
                        let poison =
                            Arc::new(EventPoison::new(handle, "Event was poisoned (cached)"));
                        return Ok(crate::events::LocalEventWaiter::immediate(Arc::new(
                            CompletionKind::Poisoned(poison),
                        )));
                    } else {
                        // Triggered
                        return Ok(crate::events::LocalEventWaiter::immediate(Arc::new(
                            CompletionKind::Triggered,
                        )));
                    }
                }
            }
        }

        // TIER 2: Check active DashMap for pending or cached events
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.local.clone())))
            .clone();

        match remote_event.register_waiter(generation) {
            WaitRegistration::Ready => Ok(crate::events::LocalEventWaiter::immediate(Arc::new(
                CompletionKind::Triggered,
            ))),
            WaitRegistration::Poisoned(poison) => Ok(crate::events::LocalEventWaiter::immediate(
                Arc::new(CompletionKind::Poisoned(poison)),
            )),
            WaitRegistration::Pending(waiter) => {
                // TIER 3: Send query to remote if first time seeing this generation
                if remote_event.add_pending(generation) {
                    let system = Arc::clone(self);
                    let remote_event_clone = remote_event.clone();
                    self.tasks.spawn(async move {
                        if let Err(e) = system.send_subscribe(handle).await {
                            // Routing failed - poison the waiter immediately
                            warn!("Failed to send subscribe for {}: {}", handle, e);
                            let poison_msg = format!("Failed to subscribe to remote event: {}", e);
                            remote_event_clone.complete_generation(
                                generation,
                                CompletionKind::Poisoned(Arc::new(EventPoison::new(
                                    handle, poison_msg,
                                ))),
                            );
                        }
                    });
                }
                Ok(waiter)
            }
        }
    }

    fn poll_remote(self: &Arc<Self>, handle: EventHandle) -> Result<EventStatus> {
        let remote_event = self
            .remote_events
            .entry(crate::events::EventKey::from_handle(handle))
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.local.clone())))
            .clone();

        let status = remote_event.status_for(handle.generation());
        if status != EventStatus::Pending {
            return Ok(status);
        }

        if remote_event.add_pending(handle.generation()) {
            let system = Arc::clone(self);
            let remote_event_clone = remote_event.clone();
            self.tasks.spawn(async move {
                if let Err(e) = system.send_subscribe(handle).await {
                    // Routing failed - poison the waiter
                    warn!("Failed to send subscribe for {}: {}", handle, e);
                    let poison_msg = format!("Failed to subscribe to remote event: {}", e);
                    remote_event_clone.complete_generation(
                        handle.generation(),
                        CompletionKind::Poisoned(Arc::new(EventPoison::new(handle, poison_msg))),
                    );
                }
            });
        }

        Ok(EventStatus::Pending)
    }

    fn trigger_remote(self: &Arc<Self>, handle: EventHandle) -> Result<ResponseAwaiter> {
        let key = crate::events::EventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache first
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key) {
                if generation > info.highest_generation {
                    // Not yet complete - fallback to TIER 2/3
                } else if generation < info.highest_generation.saturating_sub(10) {
                    // Too old (not in recent 10) - fallback to TIER 2/3 for accurate data
                } else {
                    // Within recent 10 generations - can answer from cache
                    let awaiter = self.response_manager.register_outcome()?;
                    let result = if info.poisoned_generations.contains(&generation) {
                        Err("Event was poisoned (cached)".to_string())
                    } else {
                        Ok(None) // Triggered
                    };
                    self.response_manager
                        .complete_outcome(awaiter.response_id(), result);
                    return Ok(awaiter);
                }
            }
        }

        // TIER 2: Check active DashMap
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.local.clone())))
            .clone();

        match remote_event.register_waiter(generation) {
            WaitRegistration::Ready => {
                let awaiter = self.response_manager.register_outcome()?;
                self.response_manager
                    .complete_outcome(awaiter.response_id(), Ok(None));
                Ok(awaiter)
            }
            WaitRegistration::Poisoned(poison) => {
                let awaiter = self.response_manager.register_outcome()?;
                self.response_manager
                    .complete_outcome(awaiter.response_id(), Err((*poison).reason().to_string()));
                Ok(awaiter)
            }
            WaitRegistration::Pending(_waiter) => {
                // TIER 3: Send query to remote
                let awaiter = self.response_manager.register_outcome()?;
                let response_id = awaiter.response_id();

                if remote_event.add_pending(generation) {
                    let system = Arc::clone(self);
                    let remote_event_clone = remote_event.clone();
                    self.tasks.spawn(async move {
                        if let Err(e) = system
                            .send_completion_request(handle, None, Some(response_id))
                            .await
                        {
                            // Routing failed - complete the awaiter with error and poison the waiter
                            warn!("Failed to send trigger request for {}: {}", handle, e);
                            let poison_msg = format!("Failed to send trigger request: {}", e);
                            system
                                .response_manager
                                .complete_outcome(response_id, Err(poison_msg.clone()));
                            remote_event_clone.complete_generation(
                                generation,
                                CompletionKind::Poisoned(Arc::new(EventPoison::new(
                                    handle, poison_msg,
                                ))),
                            );
                        }
                    });
                }

                Ok(awaiter)
            }
        }
    }

    fn poison_remote(
        self: &Arc<Self>,
        handle: EventHandle,
        reason: String,
    ) -> Result<ResponseAwaiter> {
        let key = crate::events::EventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache first
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key) {
                if generation > info.highest_generation {
                    // Not yet complete - fallback to TIER 2/3
                } else if generation < info.highest_generation.saturating_sub(10) {
                    // Too old (not in recent 10) - fallback to TIER 2/3 for accurate data
                } else {
                    // Within recent 10 generations - can answer from cache
                    let awaiter = self.response_manager.register_outcome()?;
                    let result = if info.poisoned_generations.contains(&generation) {
                        Ok(None) // Already poisoned - success
                    } else {
                        Err(format!("Event {} already completed successfully", handle))
                    };
                    self.response_manager
                        .complete_outcome(awaiter.response_id(), result);
                    return Ok(awaiter);
                }
            }
        }

        // TIER 2: Check active DashMap
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.local.clone())))
            .clone();

        match remote_event.register_waiter(generation) {
            WaitRegistration::Ready => {
                let awaiter = self.response_manager.register_outcome()?;
                self.response_manager.complete_outcome(
                    awaiter.response_id(),
                    Err(format!("Event {} already completed successfully", handle)),
                );
                Ok(awaiter)
            }
            WaitRegistration::Poisoned(_) => {
                let awaiter = self.response_manager.register_outcome()?;
                self.response_manager
                    .complete_outcome(awaiter.response_id(), Ok(None));
                Ok(awaiter)
            }
            WaitRegistration::Pending(_waiter) => {
                // TIER 3: Send query to remote
                let awaiter = self.response_manager.register_outcome()?;
                let response_id = awaiter.response_id();

                if remote_event.add_pending(generation) {
                    let system = Arc::clone(self);
                    let remote_event_clone = remote_event.clone();
                    self.tasks.spawn(async move {
                        if let Err(e) = system
                            .send_completion_request(
                                handle,
                                Some(reason.clone()),
                                Some(response_id),
                            )
                            .await
                        {
                            // Routing failed - complete the awaiter with error and poison the waiter
                            warn!("Failed to send poison request for {}: {}", handle, e);
                            let poison_msg = format!("Failed to send poison request: {}", e);
                            system
                                .response_manager
                                .complete_outcome(response_id, Err(poison_msg.clone()));
                            remote_event_clone.complete_generation(
                                generation,
                                CompletionKind::Poisoned(Arc::new(EventPoison::new(
                                    handle, poison_msg,
                                ))),
                            );
                        }
                    });
                }

                Ok(awaiter)
            }
        }
    }

    fn register_owner_subscription(
        self: &Arc<Self>,
        handle: EventHandle,
        target: InstanceId,
    ) -> Result<()> {
        let entry = self
            .owner_subscribers
            .entry(crate::events::EventKey::from_handle(handle))
            .or_default();

        let generation = handle.generation();
        let insert = !matches!(entry.get(&target), Some(existing) if *existing >= generation);

        if insert {
            entry.insert(target, generation);
            let system = Arc::clone(self);
            self.tasks.spawn(async move {
                let completion = match system.local.awaiter(handle) {
                    Ok(waiter) => waiter.await,
                    Err(err) => Err(err),
                };

                let completion_kind = match completion {
                    Ok(()) => CompletionKind::Triggered,
                    Err(err) => match err.downcast::<EventPoison>() {
                        Ok(poison) => CompletionKind::Poisoned(Arc::new(poison)),
                        Err(other) => CompletionKind::Poisoned(Arc::new(EventPoison::new(
                            handle,
                            other.to_string(),
                        ))),
                    },
                };

                match system.send_completion(handle, target, completion_kind) {
                    Ok(()) => {
                        // Clean up subscription entry after successful notification
                        system.cleanup_owner_subscription(handle, target);
                    }
                    Err(e) => warn!("Failed to send completion for {}: {}", handle, e),
                }
            });
        }
        Ok(())
    }

    async fn send_subscribe(&self, handle: EventHandle) -> Result<()> {
        let target_instance = self.resolve_instance_id(handle.owner_worker()).await?;

        let message = EventSubscribeMessage {
            handle: handle.raw(),
            subscriber_worker: self.worker_id.as_u64(),
            subscriber_instance: self.instance_id,
        };

        self.send_system_message(target_instance, "_event_subscribe", message)
    }

    fn send_completion(
        &self,
        handle: EventHandle,
        target: InstanceId,
        completion: CompletionKind,
    ) -> Result<()> {
        let poisoned = match &completion {
            CompletionKind::Poisoned(p) => Some(p.reason().to_string()),
            _ => None,
        };
        let message = EventCompletionMessage {
            handle: handle.raw(),
            poisoned,
        };
        self.send_system_message(target, "_event_trigger", message)
    }

    async fn send_completion_request(
        &self,
        handle: EventHandle,
        poisoned: Option<String>,
        response_id: Option<ResponseId>,
    ) -> Result<()> {
        let target_instance = self.resolve_instance_id(handle.owner_worker()).await?;

        let message = EventTriggerRequestMessage {
            handle: handle.raw(),
            requester_worker: self.worker_id.as_u64(),
            requester_instance: self.instance_id,
            poisoned,
            response_id: response_id.map(|r| r.as_u128()),
        };
        self.send_system_message(target_instance, "_event_trigger_request", message)
    }

    fn send_system_message<T: Serialize>(
        &self,
        target: InstanceId,
        handler: &str,
        payload: T,
    ) -> Result<()> {
        let messenger = self
            .messenger
            .read()
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("Event messenger not initialized"))?;
        let bytes = Bytes::from(serde_json::to_vec(&payload)?);
        messenger.send_system(target, handler, bytes)
    }

    /// Check if a RemoteEvent can be moved to the LRU cache (all pending generations complete).
    ///
    /// This is the primary memory cleanup mechanism for remote subscriptions. When a RemoteEvent
    /// has no more active waiters (`local_events` is empty) and no pending subscription requests
    /// (`pending` is empty), it means all generations have been resolved and notified.
    ///
    /// At this point, we migrate the completion history to the LRU cache for fast-path lookups
    /// and remove the RemoteEvent from the active `remote_events` DashMap to free memory.
    ///
    /// Called from:
    /// - `handle_trigger()` after receiving a completion notification from the remote owner
    ///
    /// # Arguments
    /// * `key` - The event key for the remote event
    /// * `remote_event` - Arc to the RemoteEvent to potentially cache
    fn maybe_cache_remote_event(
        &self,
        key: crate::events::EventKey,
        remote_event: Arc<RemoteEvent>,
    ) {
        // Check if all pending generations are complete
        let can_cache = {
            let pending = remote_event.pending.lock();
            let local_events = remote_event.local_events.lock();
            pending.is_empty() && local_events.is_empty()
        };

        if can_cache {
            // Move to LRU cache - just index poisoned generations
            let info = {
                let completions = remote_event.completions.lock();

                // Build index of ALL poisoned generations (without error strings to save memory)
                let poisoned_generations: std::collections::BTreeSet<u32> = completions
                    .iter()
                    .filter_map(|(generation, completion)| {
                        matches!(completion.as_ref(), CompletionKind::Poisoned(_))
                            .then_some(*generation)
                    })
                    .collect();

                CompletedEventInfo {
                    highest_generation: remote_event.known_generation(),
                    poisoned_generations,
                }
            };

            self.completed_cache.lock().put(key, info);

            // Remove from active map
            self.remote_events.remove(&key);
        }
    }

    /// Clean up owner_subscriber entry after a spawned task completes and sends notification.
    ///
    /// This is the memory cleanup mechanism for the `owner_subscribers` DashMap. When a
    /// spawned notification task successfully sends a completion message to a remote subscriber,
    /// we remove that (event, subscriber) pair from the registry.
    ///
    /// If this was the last subscriber for the event, we remove the entire EventKey entry
    /// from the DashMap to prevent unbounded memory growth.
    ///
    /// Called from:
    /// - Spawned task in `register_owner_subscription()` after successful `send_completion()`
    ///
    /// # Arguments
    /// * `handle` - The event handle for the locally-owned event
    /// * `target` - The InstanceId of the remote subscriber that was notified
    fn cleanup_owner_subscription(&self, handle: EventHandle, target: InstanceId) {
        let key = crate::events::EventKey::from_handle(handle);
        if let Some(entry) = self.owner_subscribers.get(&key) {
            entry.remove(&target);

            // If no more subscribers for this event, remove the entry
            if entry.is_empty() {
                drop(entry);
                self.owner_subscribers.remove(&key);
            }
        }
    }

    /// Send an ACK response back to the requester
    async fn send_ack(&self, response_id: ResponseId) -> Result<()> {
        let header = encode_event_header(EventType::Ack(response_id, Outcome::Ok));

        self.backend.send_message_to_worker(
            WorkerId::from_u64(response_id.worker_id()),
            header.to_vec(),
            vec![],
            MessageType::Ack,
            get_event_ack_error_handler(),
        )?;

        Ok(())
    }

    /// Send a NACK response with error message back to the requester
    async fn send_nack(&self, response_id: ResponseId, error_message: String) -> Result<()> {
        let header = encode_event_header(EventType::Ack(response_id, Outcome::Error));
        let payload = Bytes::from(error_message.into_bytes());

        self.backend.send_message_to_worker(
            WorkerId::from_u64(response_id.worker_id()),
            header.to_vec(),
            payload.to_vec(),
            MessageType::Ack,
            get_event_nack_error_handler(),
        )?;

        Ok(())
    }
}

// Error handlers for event system responses
struct EventAckErrorHandler;
impl dynamo_nova_backend::TransportErrorHandler for EventAckErrorHandler {
    fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
        warn!("Failed to send event ACK: {}", error);
    }
}

struct EventNackErrorHandler;
impl dynamo_nova_backend::TransportErrorHandler for EventNackErrorHandler {
    fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
        warn!("Failed to send event NACK: {}", error);
    }
}

static EVENT_ACK_ERROR_HANDLER: std::sync::OnceLock<
    Arc<dyn dynamo_nova_backend::TransportErrorHandler>,
> = std::sync::OnceLock::new();
static EVENT_NACK_ERROR_HANDLER: std::sync::OnceLock<
    Arc<dyn dynamo_nova_backend::TransportErrorHandler>,
> = std::sync::OnceLock::new();

fn get_event_ack_error_handler() -> Arc<dyn dynamo_nova_backend::TransportErrorHandler> {
    EVENT_ACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(EventAckErrorHandler))
        .clone()
}

fn get_event_nack_error_handler() -> Arc<dyn dynamo_nova_backend::TransportErrorHandler> {
    EVENT_NACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(EventNackErrorHandler))
        .clone()
}
