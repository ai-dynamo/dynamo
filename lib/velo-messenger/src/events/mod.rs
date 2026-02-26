// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed event system integrated into the messenger layer.
//!
//! Extends local events from `velo-events` with remote capabilities,
//! allowing events owned by one instance to be waited on, triggered,
//! or poisoned from other instances across the network.

pub(crate) mod backend;
pub(crate) mod handlers;
pub(crate) mod messages;
pub(crate) mod remote;

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use dashmap::DashMap;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio_util::task::TaskTracker;
use tracing::warn;

use velo_backend::VeloBackend;
use velo_common::{InstanceId, PeerInfo, WorkerId};
use velo_events::{
    EventAwaiter, EventHandle, EventManager, EventPoison, EventStatus, EventSystemBase,
};

use crate::common::events::{EventType, Outcome, encode_event_header};
use crate::common::responses::{ResponseAwaiter, ResponseId, ResponseManager};
use crate::messenger::Messenger;

use self::messages::{EventCompletionMessage, EventSubscribeMessage, EventTriggerRequestMessage};
use self::remote::{
    CompletedEventInfo, CompletionKind, RemoteEvent, RemoteEventKey, WaitRegistration,
};

/// Distributed event system extending velo-events with network routing.
///
/// Uses a 3-tier lookup pattern for remote event operations:
/// 1. **LRU cache** — fast path for recently completed events
/// 2. **Active DashMap** — pending subscription exists
/// 3. **Network** — send subscription request to remote owner
pub struct VeloEvents {
    local_base: Arc<EventSystemBase>,
    /// Local event manager for operations like poll that go through the manager API
    local_manager: EventManager,
    /// Event manager for proxy events that remote waiters subscribe to
    proxy_manager: EventManager,
    system_id: u64,
    instance_id: InstanceId,
    backend: Arc<VeloBackend>,
    messenger: RwLock<Option<Arc<Messenger>>>,
    remote_events: DashMap<RemoteEventKey, Arc<RemoteEvent>>,
    completed_cache: Arc<Mutex<LruCache<RemoteEventKey, CompletedEventInfo>>>,
    owner_subscribers: DashMap<RemoteEventKey, DashMap<InstanceId, u32>>,
    response_manager: ResponseManager,
    tasks: TaskTracker,
}

impl VeloEvents {
    pub(crate) fn new(
        instance_id: InstanceId,
        local_base: Arc<EventSystemBase>,
        backend: Arc<VeloBackend>,
        response_manager: ResponseManager,
    ) -> Arc<Self> {
        const DEFAULT_COMPLETED_CACHE_SIZE: usize = 1000;

        let system_id = instance_id.worker_id().as_u64();
        assert_eq!(
            system_id,
            local_base.system_id(),
            "system_id must match local event system: {} != {}",
            system_id,
            local_base.system_id()
        );
        assert_ne!(system_id, 0, "system_id must be non-zero");

        // Local manager wraps the local_base for operations like poll()
        let local_manager = EventManager::new(
            local_base.clone(),
            local_base.clone() as Arc<dyn velo_events::EventBackend>,
        );

        // Proxy manager uses a separate local system for creating proxy events
        let proxy_base = EventSystemBase::local();
        let proxy_manager = EventManager::new(
            proxy_base.clone(),
            proxy_base.clone() as Arc<dyn velo_events::EventBackend>,
        );

        Arc::new(Self {
            local_base,
            local_manager,
            proxy_manager,
            system_id,
            instance_id,
            backend,
            messenger: RwLock::new(None),
            remote_events: DashMap::new(),
            completed_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(DEFAULT_COMPLETED_CACHE_SIZE).unwrap(),
            ))),
            owner_subscribers: DashMap::new(),
            response_manager,
            tasks: TaskTracker::new(),
        })
    }

    /// Get the local event system base.
    pub fn local_base(&self) -> &Arc<EventSystemBase> {
        &self.local_base
    }

    /// Create an EventManager wired with the distributed backend.
    pub fn event_manager(self: &Arc<Self>) -> EventManager {
        let backend = Arc::new(backend::DistributedEventBackend {
            events: Arc::clone(self),
        });
        EventManager::new(self.local_base.clone(), backend)
    }

    /// Convenience: create a new event via the distributed backend.
    pub fn new_event(self: &Arc<Self>) -> Result<velo_events::Event> {
        self.event_manager().new_event()
    }

    pub(crate) fn set_messenger(&self, messenger: Arc<Messenger>) {
        *self.messenger.write() = Some(messenger);
    }

    // ── Fire-and-forget remote operations (for EventBackend trait) ──

    /// Trigger a remote event without waiting for response.
    /// Spawns a task to send the trigger request over the network.
    fn trigger_remote_fire_and_forget(self: &Arc<Self>, handle: EventHandle) -> Result<()> {
        let system = Arc::clone(self);
        self.tasks.spawn(async move {
            if let Err(e) = system.send_completion_request(handle, None, None).await {
                warn!(
                    "Failed to send fire-and-forget trigger for {}: {}",
                    handle, e
                );
            }
        });
        Ok(())
    }

    /// Poison a remote event without waiting for response.
    fn poison_remote_fire_and_forget(
        self: &Arc<Self>,
        handle: EventHandle,
        reason: Arc<str>,
    ) -> Result<()> {
        let system = Arc::clone(self);
        self.tasks.spawn(async move {
            if let Err(e) = system
                .send_completion_request(handle, Some(reason.to_string()), None)
                .await
            {
                warn!(
                    "Failed to send fire-and-forget poison for {}: {}",
                    handle, e
                );
            }
        });
        Ok(())
    }

    // ── Resolve instance_id from worker_id ──────────────────────────

    async fn resolve_instance_id(&self, worker_id: WorkerId) -> Result<InstanceId> {
        // Fast path: backend worker cache
        if let Ok(instance_id) = self.backend.try_translate_worker_id(worker_id) {
            return Ok(instance_id);
        }

        // Slow path: use messenger's discovery to resolve worker_id -> peer_info
        let messenger = self
            .messenger
            .read()
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow!("Event messenger not initialized"))?;

        let discovery = messenger.discovery().ok_or_else(|| {
            anyhow!(
                "No discovery backend configured. Cannot resolve worker {}",
                worker_id
            )
        })?;

        let peer_info: PeerInfo = discovery.discover_by_worker_id(worker_id).await?;
        let instance_id = peer_info.instance_id();

        // Auto-register discovered peer for future fast-path lookups
        messenger.register_peer(peer_info)?;

        Ok(instance_id)
    }

    // ── Public distributed operations ───────────────────────────────

    /// Create an awaiter for a handle, routing to local or remote as needed.
    pub fn awaiter(self: &Arc<Self>, handle: EventHandle) -> Result<EventAwaiter> {
        assert!(
            handle.is_distributed(),
            "Local-only events cannot be used as distributed events"
        );
        if handle.system_id() == self.system_id {
            self.local_base.awaiter_inner(handle)
        } else {
            self.wait_remote(handle)
        }
    }

    /// Poll event status, routing to local or remote as needed.
    pub fn poll(self: &Arc<Self>, handle: EventHandle) -> Result<EventStatus> {
        assert!(
            handle.is_distributed(),
            "Local-only events cannot be used as distributed events"
        );
        if handle.system_id() == self.system_id {
            self.local_manager.poll(handle)
        } else {
            self.poll_remote(handle)
        }
    }

    /// Trigger an event, routing to local or remote as needed.
    pub async fn trigger(self: &Arc<Self>, handle: EventHandle) -> Result<()> {
        assert!(
            handle.is_distributed(),
            "Local-only events cannot be used as distributed events"
        );
        if handle.system_id() == self.system_id {
            self.local_base.trigger_inner(handle)
        } else {
            let mut awaiter = self.trigger_remote(handle)?;
            awaiter.recv().await.map_err(|e| anyhow!("{}", e))?;
            Ok(())
        }
    }

    /// Poison an event, routing to local or remote as needed.
    pub async fn poison(
        self: &Arc<Self>,
        handle: EventHandle,
        reason: impl Into<String>,
    ) -> Result<()> {
        assert!(
            handle.is_distributed(),
            "Local-only events cannot be used as distributed events"
        );
        if handle.system_id() == self.system_id {
            self.local_base
                .poison_inner(handle, Arc::<str>::from(reason.into()))
        } else {
            let mut awaiter = self.poison_remote(handle, reason.into())?;
            awaiter.recv().await.map_err(|e| anyhow!("{}", e))?;
            Ok(())
        }
    }

    // ── Owner-side handlers ─────────────────────────────────────────

    pub(crate) fn handle_subscribe(self: &Arc<Self>, payload: Bytes) -> Result<()> {
        let message: EventSubscribeMessage = serde_json::from_slice(&payload)?;
        let handle = EventHandle::from_raw(message.handle);
        if handle.system_id() != self.system_id {
            bail!("Subscribe for non-local event {}", handle);
        }

        match self.local_manager.poll(handle)? {
            EventStatus::Ready => self.send_completion(
                handle,
                message.subscriber_instance,
                CompletionKind::Triggered,
            ),
            EventStatus::Poisoned => {
                let reason = "event was poisoned".to_string();
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

        let key = RemoteEventKey::from_handle(handle);
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.proxy_manager.clone())))
            .clone();

        remote_event.complete_generation(handle.generation(), completion);
        self.maybe_cache_remote_event(key, remote_event);

        Ok(())
    }

    pub(crate) fn handle_trigger_request(self: &Arc<Self>, payload: Bytes) -> Result<()> {
        let message: EventTriggerRequestMessage = serde_json::from_slice(&payload)?;
        let handle = EventHandle::from_raw(message.handle);
        if handle.system_id() != self.system_id {
            bail!("Trigger request for non-local event {}", handle);
        }

        if let Some(response_id_raw) = message.response_id {
            let response_id = ResponseId::from_u128(response_id_raw);

            match self.local_manager.poll(handle)? {
                EventStatus::Ready | EventStatus::Poisoned => {
                    let completion = match self.local_manager.poll(handle)? {
                        EventStatus::Poisoned => {
                            let reason = "event was poisoned".to_string();
                            CompletionKind::Poisoned(Arc::new(EventPoison::new(handle, reason)))
                        }
                        _ => CompletionKind::Triggered,
                    };
                    self.send_completion(handle, message.requester_instance, completion)?;

                    let system = Arc::clone(self);
                    self.tasks.spawn(async move {
                        if system.local_manager.poll(handle).ok() == Some(EventStatus::Poisoned) {
                            let reason = "event was poisoned".to_string();
                            let _ = system.send_nack(response_id, reason).await;
                        } else {
                            let _ = system.send_ack(response_id).await;
                        }
                    });
                    Ok(())
                }
                EventStatus::Pending => {
                    self.register_owner_subscription(handle, message.requester_instance)?;

                    let is_poison_request = message.poisoned.is_some();

                    if let Some(reason) = message.poisoned {
                        self.local_base
                            .poison_inner(handle, Arc::<str>::from(reason))?;
                    } else {
                        self.local_base.trigger_inner(handle)?;
                    }

                    let system = Arc::clone(self);
                    self.tasks.spawn(async move {
                        match system.local_base.awaiter_inner(handle) {
                            Ok(waiter) => match waiter.await {
                                Ok(()) => {
                                    let _ = system.send_ack(response_id).await;
                                }
                                Err(poison_err) => {
                                    if is_poison_request {
                                        let _ = system.send_ack(response_id).await;
                                    } else {
                                        let _ = system
                                            .send_nack(response_id, poison_err.to_string())
                                            .await;
                                    }
                                }
                            },
                            Err(e) => {
                                let _ = system.send_nack(response_id, e.to_string()).await;
                            }
                        }
                    });
                    Ok(())
                }
            }
        } else {
            // Legacy path without response_id
            match self.local_manager.poll(handle)? {
                EventStatus::Ready => self.send_completion(
                    handle,
                    message.requester_instance,
                    CompletionKind::Triggered,
                ),
                EventStatus::Poisoned => {
                    let reason = "event was poisoned".to_string();
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
                        self.local_base
                            .poison_inner(handle, Arc::<str>::from(reason))?;
                    } else {
                        self.local_base.trigger_inner(handle)?;
                    }
                    Ok(())
                }
            }
        }
    }

    // ── Subscriber-side paths ───────────────────────────────────────

    fn wait_remote(self: &Arc<Self>, handle: EventHandle) -> Result<EventAwaiter> {
        let key = RemoteEventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key)
                && generation <= info.highest_generation
                && generation >= info.highest_generation.saturating_sub(10)
            {
                if info.poisoned_generations.contains(&generation) {
                    let poison = Arc::new(EventPoison::new(handle, "Event was poisoned (cached)"));
                    return self.immediate_poison_awaiter(poison);
                } else {
                    return self.immediate_ok_awaiter();
                }
            }
        }

        // TIER 2: Check active DashMap
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.proxy_manager.clone())))
            .clone();

        match remote_event.register_waiter(generation) {
            WaitRegistration::Ready => self.immediate_ok_awaiter(),
            WaitRegistration::Poisoned(poison) => self.immediate_poison_awaiter(poison),
            WaitRegistration::Pending(waiter) => {
                // TIER 3: Send subscription if first time
                if remote_event.add_pending(generation) {
                    let system = Arc::clone(self);
                    let remote_event_clone = remote_event.clone();
                    self.tasks.spawn(async move {
                        if let Err(e) = system.send_subscribe(handle).await {
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
        let key = RemoteEventKey::from_handle(handle);
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.proxy_manager.clone())))
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
        let key = RemoteEventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key)
                && generation <= info.highest_generation
                && generation >= info.highest_generation.saturating_sub(10)
            {
                let awaiter = self.response_manager.register_outcome()?;
                let result = if info.poisoned_generations.contains(&generation) {
                    Err("Event was poisoned (cached)".to_string())
                } else {
                    Ok(None)
                };
                self.response_manager
                    .complete_outcome(awaiter.response_id(), result);
                return Ok(awaiter);
            }
        }

        // TIER 2: Check active DashMap
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.proxy_manager.clone())))
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
                // TIER 3: Send trigger request
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
        let key = RemoteEventKey::from_handle(handle);
        let generation = handle.generation();

        // TIER 1: Check LRU cache
        {
            let mut cache = self.completed_cache.lock();
            if let Some(info) = cache.get(&key)
                && generation <= info.highest_generation
                && generation >= info.highest_generation.saturating_sub(10)
            {
                let awaiter = self.response_manager.register_outcome()?;
                let result = if info.poisoned_generations.contains(&generation) {
                    Ok(None) // Already poisoned
                } else {
                    Err(format!("Event {} already completed successfully", handle))
                };
                self.response_manager
                    .complete_outcome(awaiter.response_id(), result);
                return Ok(awaiter);
            }
        }

        // TIER 2: Check active DashMap
        let remote_event = self
            .remote_events
            .entry(key)
            .or_insert_with(|| Arc::new(RemoteEvent::new(self.proxy_manager.clone())))
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
                // TIER 3: Send poison request
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

    // ── Owner subscription management ───────────────────────────────

    fn register_owner_subscription(
        self: &Arc<Self>,
        handle: EventHandle,
        target: InstanceId,
    ) -> Result<()> {
        let key = RemoteEventKey::from_handle(handle);
        let entry = self.owner_subscribers.entry(key).or_default();

        let generation = handle.generation();
        let insert = !matches!(entry.get(&target), Some(existing) if *existing >= generation);

        if insert {
            entry.insert(target, generation);
            let system = Arc::clone(self);
            self.tasks.spawn(async move {
                let completion = match system.local_base.awaiter_inner(handle) {
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
                    Ok(()) => system.cleanup_owner_subscription(handle, target),
                    Err(e) => warn!("Failed to send completion for {}: {}", handle, e),
                }
            });
        }
        Ok(())
    }

    // ── Messaging helpers ───────────────────────────────────────────

    async fn send_subscribe(&self, handle: EventHandle) -> Result<()> {
        let target_worker = WorkerId::from_u64(handle.system_id());
        let target_instance = self.resolve_instance_id(target_worker).await?;

        let message = EventSubscribeMessage {
            handle: handle.raw(),
            subscriber_worker: self.system_id,
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
        let target_worker = WorkerId::from_u64(handle.system_id());
        let target_instance = self.resolve_instance_id(target_worker).await?;

        let message = EventTriggerRequestMessage {
            handle: handle.raw(),
            requester_worker: self.system_id,
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
        let builder = messenger
            .message_builder_unchecked(handler)
            .raw_payload(bytes)
            .instance(target);
        self.tasks.spawn(async move {
            if let Err(e) = builder.fire().await {
                warn!("Failed to send system message: {}", e);
            }
        });
        Ok(())
    }

    // ── LRU cache management ────────────────────────────────────────

    fn maybe_cache_remote_event(&self, key: RemoteEventKey, remote_event: Arc<RemoteEvent>) {
        if remote_event.is_cacheable() {
            let info = CompletedEventInfo {
                highest_generation: remote_event.known_generation(),
                poisoned_generations: BTreeSet::new(), // simplified
            };

            self.completed_cache.lock().put(key, info);
            self.remote_events.remove(&key);
        }
    }

    fn cleanup_owner_subscription(&self, handle: EventHandle, target: InstanceId) {
        let key = RemoteEventKey::from_handle(handle);
        if let Some(entry) = self.owner_subscribers.get(&key) {
            entry.remove(&target);
            if entry.is_empty() {
                drop(entry);
                self.owner_subscribers.remove(&key);
            }
        }
    }

    // ── ACK/NACK helpers ────────────────────────────────────────────

    async fn send_ack(&self, response_id: ResponseId) -> Result<()> {
        let header = encode_event_header(EventType::Ack(response_id, Outcome::Ok));

        self.backend.send_message_to_worker(
            WorkerId::from_u64(response_id.worker_id()),
            header.to_vec(),
            vec![],
            velo_backend::MessageType::Ack,
            get_event_ack_error_handler(),
        )?;

        Ok(())
    }

    async fn send_nack(&self, response_id: ResponseId, error_message: String) -> Result<()> {
        let header = encode_event_header(EventType::Ack(response_id, Outcome::Error));
        let payload = error_message.into_bytes();

        self.backend.send_message_to_worker(
            WorkerId::from_u64(response_id.worker_id()),
            header.to_vec(),
            payload,
            velo_backend::MessageType::Ack,
            get_event_nack_error_handler(),
        )?;

        Ok(())
    }

    // ── Immediate awaiter helpers ───────────────────────────────────

    fn immediate_ok_awaiter(&self) -> Result<EventAwaiter> {
        // Create a proxy event and immediately trigger it
        let event = self.proxy_manager.new_event()?;
        let handle = event.handle();
        event.trigger()?;
        self.proxy_manager.awaiter(handle)
    }

    fn immediate_poison_awaiter(&self, poison: Arc<EventPoison>) -> Result<EventAwaiter> {
        // Create a proxy event and immediately poison it
        let event = self.proxy_manager.new_event()?;
        let handle = event.handle();
        event.poison(poison.reason())?;
        self.proxy_manager.awaiter(handle)
    }
}

// ── Error handlers for event system responses ───────────────────────

struct EventAckErrorHandler;
impl velo_backend::TransportErrorHandler for EventAckErrorHandler {
    fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
        warn!("Failed to send event ACK: {}", error);
    }
}

struct EventNackErrorHandler;
impl velo_backend::TransportErrorHandler for EventNackErrorHandler {
    fn on_error(&self, _header: bytes::Bytes, _payload: bytes::Bytes, error: String) {
        warn!("Failed to send event NACK: {}", error);
    }
}

static EVENT_ACK_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();
static EVENT_NACK_ERROR_HANDLER: std::sync::OnceLock<Arc<dyn velo_backend::TransportErrorHandler>> =
    std::sync::OnceLock::new();

fn get_event_ack_error_handler() -> Arc<dyn velo_backend::TransportErrorHandler> {
    EVENT_ACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(EventAckErrorHandler))
        .clone()
}

fn get_event_nack_error_handler() -> Arc<dyn velo_backend::TransportErrorHandler> {
    EVENT_NACK_ERROR_HANDLER
        .get_or_init(|| Arc::new(EventNackErrorHandler))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::remote::*;
    use std::sync::Arc;
    use std::time::Duration;
    use std::num::NonZero;
    use velo_events::{
        DistributedEventFactory, EventHandle, EventManager, EventPoison, EventStatus,
    };

    fn make_proxy_manager() -> EventManager {
        let system_id = NonZero::new(42u64).unwrap();
        let factory = DistributedEventFactory::new(system_id);
        let base = factory.system().clone();
        EventManager::new(base.clone(), base as Arc<dyn velo_events::EventBackend>)
    }

    // ── RemoteEvent unit tests ──────────────────────────────────────

    #[test]
    fn remote_event_register_waiter_returns_pending() {
        let re = RemoteEvent::new(make_proxy_manager());
        match re.register_waiter(1) {
            WaitRegistration::Pending(_) => {}
            other => panic!("Expected Pending, got {:?}", variant_name(&other)),
        }
    }

    #[tokio::test]
    async fn remote_event_complete_trigger_wakes_waiter() {
        let re = RemoteEvent::new(make_proxy_manager());
        let waiter = match re.register_waiter(1) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };

        re.complete_generation(1, CompletionKind::Triggered);

        let result = tokio::time::timeout(Duration::from_millis(100), waiter).await;
        assert!(result.is_ok(), "Waiter should resolve after trigger");
    }

    #[tokio::test]
    async fn remote_event_complete_poison_wakes_waiter() {
        let re = RemoteEvent::new(make_proxy_manager());
        let waiter = match re.register_waiter(1) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };

        let handle = EventHandle::from_raw(0x0001_0001_0001_0001_u128);
        let poison = Arc::new(EventPoison::new(handle, "test error"));
        re.complete_generation(1, CompletionKind::Poisoned(poison));

        let result = tokio::time::timeout(Duration::from_millis(100), waiter).await;
        assert!(result.is_ok(), "Waiter should resolve after poison");
    }

    #[test]
    fn remote_event_register_after_trigger_returns_ready() {
        let re = RemoteEvent::new(make_proxy_manager());
        re.complete_generation(1, CompletionKind::Triggered);

        match re.register_waiter(1) {
            WaitRegistration::Ready => {}
            other => panic!("Expected Ready, got {:?}", variant_name(&other)),
        }
    }

    #[test]
    fn remote_event_register_after_poison_returns_poisoned() {
        let re = RemoteEvent::new(make_proxy_manager());
        let handle = EventHandle::from_raw(0x0001_0001_0001_0001_u128);
        let poison = Arc::new(EventPoison::new(handle, "bad"));
        re.complete_generation(1, CompletionKind::Poisoned(poison));

        match re.register_waiter(1) {
            WaitRegistration::Poisoned(_) => {}
            other => panic!("Expected Poisoned, got {:?}", variant_name(&other)),
        }
    }

    #[test]
    fn remote_event_status_for_pending_ready_poisoned() {
        let re = RemoteEvent::new(make_proxy_manager());
        assert_eq!(re.status_for(1), EventStatus::Pending);

        re.complete_generation(1, CompletionKind::Triggered);
        assert_eq!(re.status_for(1), EventStatus::Ready);

        let handle = EventHandle::from_raw(0x0001_0001_0001_0001_u128);
        let poison = Arc::new(EventPoison::new(handle, "err"));
        re.complete_generation(2, CompletionKind::Poisoned(poison));
        assert_eq!(re.status_for(2), EventStatus::Poisoned);

        // Generation 3 still pending
        assert_eq!(re.status_for(3), EventStatus::Pending);
    }

    #[test]
    fn remote_event_add_pending_deduplication() {
        let re = RemoteEvent::new(make_proxy_manager());
        assert!(re.add_pending(1), "First add should return true");
        assert!(!re.add_pending(1), "Second add should return false");
        assert!(re.add_pending(2), "Different generation should return true");
    }

    #[test]
    fn remote_event_is_cacheable() {
        let re = RemoteEvent::new(make_proxy_manager());
        assert!(re.is_cacheable(), "Fresh RemoteEvent should be cacheable");

        re.add_pending(1);
        assert!(!re.is_cacheable(), "With pending should not be cacheable");
    }

    #[test]
    fn remote_event_cacheable_after_all_completed() {
        let re = RemoteEvent::new(make_proxy_manager());
        let _waiter = re.register_waiter(1);
        re.add_pending(1);

        assert!(!re.is_cacheable());

        re.complete_generation(1, CompletionKind::Triggered);
        assert!(re.is_cacheable(), "Should be cacheable after all completed");
    }

    #[tokio::test]
    async fn remote_event_higher_generation_completes_lower() {
        let re = RemoteEvent::new(make_proxy_manager());
        let w1 = match re.register_waiter(1) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };
        let w2 = match re.register_waiter(2) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };

        // Completing generation 2 should also complete generation 1
        re.complete_generation(2, CompletionKind::Triggered);

        let _ = tokio::time::timeout(Duration::from_millis(100), w1)
            .await
            .expect("Waiter 1 should resolve");
        let _ = tokio::time::timeout(Duration::from_millis(100), w2)
            .await
            .expect("Waiter 2 should resolve");
    }

    #[test]
    fn remote_event_known_generation_monotonic() {
        let re = RemoteEvent::new(make_proxy_manager());
        assert_eq!(re.known_generation(), 0);

        re.complete_generation(5, CompletionKind::Triggered);
        assert_eq!(re.known_generation(), 5);

        // Lower generation should not decrease known_generation
        re.complete_generation(3, CompletionKind::Triggered);
        assert_eq!(re.known_generation(), 5);

        re.complete_generation(10, CompletionKind::Triggered);
        assert_eq!(re.known_generation(), 10);
    }

    #[test]
    fn remote_event_completion_history_bounded() {
        let re = RemoteEvent::new(make_proxy_manager());
        // Complete 150 generations
        for i in 1..=150 {
            re.complete_generation(i, CompletionKind::Triggered);
        }

        // Old generations should be pruned (history bounded to 100)
        // Generation 1 should no longer have completion entry, but still be Ready
        // (because it's <= known_generation)
        assert_eq!(re.status_for(1), EventStatus::Ready);
        assert_eq!(re.status_for(150), EventStatus::Ready);
    }

    #[tokio::test]
    async fn remote_event_multiple_waiters_same_generation() {
        let re = RemoteEvent::new(make_proxy_manager());
        let w1 = match re.register_waiter(1) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };
        let w2 = match re.register_waiter(1) {
            WaitRegistration::Pending(w) => w,
            _ => panic!("Expected Pending"),
        };

        re.complete_generation(1, CompletionKind::Triggered);

        let _ = tokio::time::timeout(Duration::from_millis(100), w1)
            .await
            .expect("Waiter 1 should resolve");
        let _ = tokio::time::timeout(Duration::from_millis(100), w2)
            .await
            .expect("Waiter 2 should resolve");
    }

    // ── Two-messenger integration tests ─────────────────────────────

    use crate::Messenger;
    use velo_backend::tcp::TcpTransportBuilder;

    fn new_transport() -> Arc<velo_backend::tcp::TcpTransport> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        )
    }

    async fn make_pair() -> (Arc<Messenger>, Arc<Messenger>) {
        let a = Messenger::new(vec![new_transport()], None).await.unwrap();
        let b = Messenger::new(vec![new_transport()], None).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        a.register_peer(b.peer_info()).unwrap();
        b.register_peer(a.peer_info()).unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;

        (a, b)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn local_event_trigger_via_events() {
        let (a, _b) = make_pair().await;

        let em = a.event_manager();
        let event = em.new_event().unwrap();
        let handle = event.handle();

        let awaiter = a.events().awaiter(handle).unwrap();
        a.events().local_base().trigger_inner(handle).unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
        assert!(result.is_ok(), "Local event should resolve");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn local_event_poison_via_events() {
        let (a, _b) = make_pair().await;

        let em = a.event_manager();
        let event = em.new_event().unwrap();
        let handle = event.handle();

        let awaiter = a.events().awaiter(handle).unwrap();
        a.events()
            .local_base()
            .poison_inner(handle, Arc::from("test failure"))
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
        assert!(result.is_ok(), "Local poison should resolve awaiter");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn remote_subscribe_then_trigger() {
        let (a, b) = make_pair().await;

        // A creates event
        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // B subscribes
        let awaiter = b.events().awaiter(handle).unwrap();

        // Give subscription time to propagate
        tokio::time::sleep(Duration::from_millis(500)).await;

        // A triggers
        a.events().local_base().trigger_inner(handle).unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
        assert!(
            result.is_ok(),
            "Remote awaiter should resolve after trigger"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn remote_subscribe_then_poison() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        let awaiter = b.events().awaiter(handle).unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        a.events()
            .local_base()
            .poison_inner(handle, Arc::from("test error"))
            .unwrap();

        let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
        assert!(result.is_ok(), "Remote awaiter should resolve after poison");
        // The awaiter resolves with Err for poison
        assert!(result.unwrap().is_err(), "Poison should produce error");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn remote_trigger_request() {
        let (a, b) = make_pair().await;

        // A creates event (pending)
        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // B triggers A's event remotely
        b.events().trigger(handle).await.unwrap();

        // A's event should now be complete
        let status = a.events().poll(handle).unwrap();
        assert_eq!(status, EventStatus::Ready);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn remote_poison_request() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // B poisons A's event remotely
        b.events().poison(handle, "remote poison").await.unwrap();

        let status = a.events().poll(handle).unwrap();
        assert_eq!(status, EventStatus::Poisoned);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn subscribe_to_already_triggered_event() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // Trigger before anyone subscribes
        a.events().local_base().trigger_inner(handle).unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // B subscribes after trigger — should get immediate completion
        let awaiter = b.events().awaiter(handle).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
        assert!(
            result.is_ok(),
            "Subscribe to already-triggered event should resolve"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn subscribe_to_already_poisoned_event() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // Poison before anyone subscribes
        a.events()
            .local_base()
            .poison_inner(handle, Arc::from("early poison"))
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // B subscribes after poison
        let awaiter = b.events().awaiter(handle).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), awaiter).await;
        assert!(
            result.is_ok(),
            "Subscribe to already-poisoned event should resolve"
        );
        assert!(result.unwrap().is_err(), "Should receive poison error");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multiple_remote_subscribers() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // B subscribes twice to the same event
        let awaiter1 = b.events().awaiter(handle).unwrap();
        let awaiter2 = b.events().awaiter(handle).unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        a.events().local_base().trigger_inner(handle).unwrap();

        let (r1, r2) = tokio::join!(
            tokio::time::timeout(Duration::from_secs(5), awaiter1),
            tokio::time::timeout(Duration::from_secs(5), awaiter2),
        );
        assert!(r1.is_ok(), "First subscriber should resolve");
        assert!(r2.is_ok(), "Second subscriber should resolve");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn poll_remote_event_pending_before_trigger() {
        let (a, b) = make_pair().await;

        let em_a = a.event_manager();
        let event = em_a.new_event().unwrap();
        let handle = event.handle();

        // Initially pending from B's perspective
        let status = b.events().poll(handle).unwrap();
        assert_eq!(status, EventStatus::Pending);

        // Subscribe so B tracks the event
        let awaiter = b.events().awaiter(handle).unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        a.events().local_base().trigger_inner(handle).unwrap();

        // Awaiter resolves
        tokio::time::timeout(Duration::from_secs(5), awaiter)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_backend_routes_local_trigger() {
        let (a, _b) = make_pair().await;

        // Use the EventManager (which uses DistributedEventBackend)
        let em = a.event_manager();
        let event = em.new_event().unwrap();
        let handle = event.handle();

        let awaiter = em.awaiter(handle).unwrap();
        em.trigger(handle).unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
        assert!(
            result.is_ok(),
            "DistributedEventBackend local trigger should work"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_backend_routes_local_poison() {
        let (a, _b) = make_pair().await;

        let em = a.event_manager();
        let event = em.new_event().unwrap();
        let handle = event.handle();

        let awaiter = em.awaiter(handle).unwrap();
        em.poison(handle, "backend test").unwrap();

        let result = tokio::time::timeout(Duration::from_secs(2), awaiter).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_err(), "Poison should produce error");
    }

    fn variant_name(wr: &WaitRegistration) -> &'static str {
        match wr {
            WaitRegistration::Ready => "Ready",
            WaitRegistration::Pending(_) => "Pending",
            WaitRegistration::Poisoned(_) => "Poisoned",
        }
    }
}
