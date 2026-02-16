// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex as ParkingMutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tokio_util::task::TaskTracker;
use tracing::{error, trace};

use super::event::LocalEvent;
use crate::event::Event;
use crate::handle::{EventHandle, LOCAL_FLAG};
use crate::manager::EventManager;
use crate::slot::{
    CompletionKind, EventAwaiter, EventEntry, EventKey, PoisonArc, WaitRegistration,
};
use crate::status::{EventPoison, EventStatus};

/// Maximum counter value for local indices (31-bit counter space, ~2B entries).
const MAX_LOCAL_INDEX: u32 = (1u32 << 31) - 1;

/// Local event system with reusable event entries.
///
/// Events created by a `LocalEventSystem` are bound to that system. Passing
/// a handle from one local system to another will return an error. For
/// cross-system event coordination, use a distributed event system that
/// implements [`EventManager`] with routing logic for remote handles.
pub struct LocalEventSystem {
    system_id: u64,
    is_local: bool,
    events: DashMap<EventKey, Arc<EventEntry>>,
    free_lists: ParkingMutex<VecDeque<Arc<EventEntry>>>,
    next_local_index: AtomicU32,
    tasks: TaskTracker,
    shutdown: AtomicBool,
}

impl LocalEventSystem {
    /// Create a new local event system with a random system_id.
    ///
    /// The system_id is derived from `xxh3_64(Uuid::new_v4())` to ensure
    /// each local system is uniquely identifiable. Handles produced by this
    /// system have bit 31 set in their `local_index` to mark them as local.
    ///
    /// Events created by this system can only be triggered, awaited, poisoned,
    /// or polled through this same system instance.
    pub fn new() -> Arc<Self> {
        let system_id = xxhash_rust::xxh3::xxh3_64(uuid::Uuid::new_v4().as_bytes());
        Self::create(system_id, true)
    }

    /// Create a system pre-configured with a system_id for distributed use.
    pub(crate) fn with_system_id(system_id: u64) -> Arc<Self> {
        Self::create(system_id, false)
    }

    fn create(system_id: u64, is_local: bool) -> Arc<Self> {
        Arc::new(Self {
            system_id,
            is_local,
            events: DashMap::new(),
            free_lists: ParkingMutex::new(VecDeque::new()),
            next_local_index: AtomicU32::new(0),
            tasks: TaskTracker::new(),
            shutdown: AtomicBool::new(false),
        })
    }

    pub fn system_id(&self) -> u64 {
        self.system_id
    }

    pub fn task_tracker(&self) -> &TaskTracker {
        &self.tasks
    }

    // ── Ownership validation ─────────────────────────────────────────

    fn validate_handle(&self, handle: EventHandle) -> Result<()> {
        if handle.system_id() != self.system_id {
            bail!(
                "Handle {} belongs to system {:#x}, not this system {:#x}",
                handle,
                handle.system_id(),
                self.system_id,
            );
        }
        Ok(())
    }

    // ── Internal helpers ──────────────────────────────────────────────

    fn new_event_inner(self: &Arc<Self>) -> Result<LocalEvent> {
        if self.is_shutdown() {
            bail!("Event system shutdown in progress");
        }
        loop {
            let entry = self.allocate_entry()?;
            match entry.begin_generation() {
                Ok(generation) => {
                    if self.is_shutdown() {
                        let handle = entry.key().handle(self.system_id, generation);
                        let poison = Arc::new(EventPoison::new(
                            handle,
                            "Event system shutdown in progress",
                        ));
                        let _ = self.poison_local_entry(entry, handle, poison);
                        bail!("Event system shutdown in progress");
                    }
                    let handle = entry.key().handle(self.system_id, generation);
                    return Ok(LocalEvent::new(self.clone(), entry, handle));
                }
                Err(crate::slot::entry::EventEntryError::GenerationOverflow { key }) => {
                    trace!(
                        ?key,
                        "retiring event entry after exhausting generation space"
                    );
                    self.retire_entry(entry);
                    continue;
                }
                Err(err) => {
                    self.recycle_entry(entry);
                    return Err(err.into());
                }
            }
        }
    }

    pub(crate) fn awaiter_inner(&self, handle: EventHandle) -> Result<EventAwaiter> {
        self.validate_handle(handle)?;
        self.wait_local(handle)
    }

    fn poll_inner(&self, handle: EventHandle) -> Result<EventStatus> {
        self.validate_handle(handle)?;
        self.poll_local(handle)
    }

    fn trigger_inner(&self, handle: EventHandle) -> Result<()> {
        self.validate_handle(handle)?;
        let entry = self
            .events
            .get(&EventKey::from_handle(handle))
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown event {}", handle))?;

        self.trigger_local_entry(entry, handle)
    }

    fn poison_inner(&self, handle: EventHandle, reason: impl Into<Arc<str>>) -> Result<()> {
        self.validate_handle(handle)?;
        let reason: Arc<str> = reason.into();

        let entry = self
            .events
            .get(&EventKey::from_handle(handle))
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown event {}", handle))?;

        match entry.status_for(handle.generation()) {
            EventStatus::Poisoned => return Ok(()),
            EventStatus::Ready => {
                bail!("Event {} already completed successfully", handle);
            }
            EventStatus::Pending => {}
        }

        let poison = Arc::new(EventPoison::new(handle, reason));
        self.poison_local_entry(entry, handle, poison)
    }

    fn merge_events_inner(self: &Arc<Self>, inputs: Vec<EventHandle>) -> Result<EventHandle> {
        if inputs.is_empty() {
            bail!("Cannot merge empty event list");
        }

        for input in &inputs {
            self.validate_handle(*input)?;
        }

        let merged = self.new_event_inner()?;
        let handle = merged.handle();
        let trigger = merged.clone();

        let system = Arc::clone(self);
        self.tasks.spawn(async move {
            let mut failure_reasons: Option<Vec<Arc<str>>> = None;

            for dependency in &inputs {
                let wait_result = match system.awaiter_inner(*dependency) {
                    Ok(waiter) => waiter.await,
                    Err(err) => Err(err),
                };

                match wait_result {
                    Ok(()) => {}
                    Err(err) => {
                        let reason = match err.downcast::<EventPoison>() {
                            Ok(poison) => format!(
                                "Merge dependency {} poisoned: {}",
                                dependency,
                                poison.reason()
                            ),
                            Err(other) => {
                                format!("Merge dependency {} failed: {}", dependency, other)
                            }
                        };
                        let reason_arc: Arc<str> = Arc::from(reason);
                        error!("{}", &*reason_arc);
                        failure_reasons
                            .get_or_insert_with(Vec::new)
                            .push(reason_arc);
                    }
                }
            }

            let result = match failure_reasons {
                None => trigger.trigger(),
                Some(reasons) => {
                    if reasons.len() == 1 {
                        system.poison_inner(handle, reasons[0].clone())
                    } else {
                        let mut message = String::from("Multiple merge dependencies failed:\n");
                        for (idx, reason) in reasons.iter().enumerate() {
                            if idx > 0 {
                                message.push('\n');
                            }
                            message.push_str(reason.as_ref());
                        }
                        system.poison_inner(handle, message)
                    }
                }
            };

            if let Err(e) = result {
                error!("Failed to complete merged event {}: {}", handle, e);
            }
        });

        Ok(handle)
    }

    fn force_shutdown_inner(&self, reason: impl Into<Arc<str>>) {
        let was_shutdown = self.shutdown.swap(true, Ordering::SeqCst);
        if was_shutdown {
            return;
        }

        let reason: Arc<str> = reason.into();

        let mut pending = Vec::new();
        for entry in self.events.iter() {
            if let Some(handle) = entry.value().active_handle(self.system_id) {
                pending.push((entry.value().clone(), handle));
            }
        }

        for (entry, handle) in pending {
            let poison = Arc::new(EventPoison::new(handle, Arc::clone(&reason)));
            if let Err(err) = self.poison_local_entry(entry, handle, poison) {
                error!("force_shutdown: failed to poison {}: {}", handle, err);
            }
        }

        self.free_lists.lock().clear();
    }

    // ── Low-level helpers ─────────────────────────────────────────────

    /// Return the poison reason for a completed generation, if any.
    #[allow(dead_code)]
    pub(crate) fn poison_reason(&self, handle: EventHandle) -> Option<Arc<str>> {
        let entry = self.events.get(&EventKey::from_handle(handle))?;
        entry.poison_reason(handle.generation())
    }

    pub(crate) fn trigger_local_entry(
        &self,
        entry: Arc<EventEntry>,
        handle: EventHandle,
    ) -> Result<()> {
        self.complete_local_entry(entry, handle, CompletionKind::Triggered)
    }

    pub(crate) fn poison_local_entry(
        &self,
        entry: Arc<EventEntry>,
        handle: EventHandle,
        poison: PoisonArc,
    ) -> Result<()> {
        self.complete_local_entry(entry, handle, CompletionKind::Poisoned(poison))
    }

    fn complete_local_entry(
        &self,
        entry: Arc<EventEntry>,
        handle: EventHandle,
        completion: CompletionKind,
    ) -> Result<()> {
        entry
            .finalize_completion(handle.generation(), completion)
            .map_err(anyhow::Error::new)?;
        self.recycle_entry(entry);
        Ok(())
    }

    fn wait_local(&self, handle: EventHandle) -> Result<EventAwaiter> {
        let entry = self
            .events
            .get(&EventKey::from_handle(handle))
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown local event {}", handle))?;

        match entry.register_local_waiter(handle.generation())? {
            WaitRegistration::Ready => {
                Ok(EventAwaiter::immediate(Arc::new(CompletionKind::Triggered)))
            }
            WaitRegistration::Poisoned(poison) => Ok(EventAwaiter::immediate(Arc::new(
                CompletionKind::Poisoned(poison),
            ))),
            WaitRegistration::Pending => Ok(EventAwaiter::pending(entry, handle.generation())),
        }
    }

    fn poll_local(&self, handle: EventHandle) -> Result<EventStatus> {
        let entry = self
            .events
            .get(&EventKey::from_handle(handle))
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown local event {}", handle))?;

        Ok(entry.status_for(handle.generation()))
    }

    fn allocate_entry(self: &Arc<Self>) -> Result<Arc<EventEntry>> {
        if let Some(entry) = self.try_reuse_entry() {
            return Ok(entry);
        }

        let counter = self
            .next_local_index
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < MAX_LOCAL_INDEX).then_some(current + 1)
            })
            .map_err(|_| {
                anyhow!(
                    "Local event index space exhausted ({} entries)",
                    (MAX_LOCAL_INDEX as u64) + 1
                )
            })?;

        let local_index = if self.is_local {
            counter | LOCAL_FLAG
        } else {
            counter
        };

        let key = EventKey::new(local_index);
        let entry = Arc::new(EventEntry::new(key));
        self.events.insert(key, entry.clone());
        Ok(entry)
    }

    fn try_reuse_entry(&self) -> Option<Arc<EventEntry>> {
        let mut free_lists = self.free_lists.lock();
        free_lists.pop_front()
    }

    fn recycle_entry(&self, entry: Arc<EventEntry>) {
        if entry.is_retired() {
            return;
        }
        let mut free_lists = self.free_lists.lock();
        free_lists.push_back(entry);
    }

    fn retire_entry(&self, entry: Arc<EventEntry>) {
        entry.retire();
    }

    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl EventManager for Arc<LocalEventSystem> {
    type Event = LocalEvent;

    fn new_event(&self) -> Result<LocalEvent> {
        self.new_event_inner()
    }

    fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter> {
        self.awaiter_inner(handle)
    }

    fn poll(&self, handle: EventHandle) -> Result<EventStatus> {
        self.poll_inner(handle)
    }

    fn trigger(&self, handle: EventHandle) -> Result<()> {
        self.trigger_inner(handle)
    }

    fn poison(&self, handle: EventHandle, reason: impl Into<Arc<str>>) -> Result<()> {
        self.poison_inner(handle, reason)
    }

    fn merge_events(&self, inputs: Vec<EventHandle>) -> Result<EventHandle> {
        self.merge_events_inner(inputs)
    }

    fn force_shutdown(&self, reason: impl Into<Arc<str>>) {
        self.force_shutdown_inner(reason)
    }
}
