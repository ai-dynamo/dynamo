// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete [`Event`] RAII guard and [`EventBackend`] routing trait.

use anyhow::Result;
use std::sync::{Arc, LazyLock};

use crate::handle::EventHandle;
use crate::slot::EventAwaiter;

/// Static poison reason reused across all drop-triggered poisons.
static DROP_POISON_REASON: LazyLock<Arc<str>> =
    LazyLock::new(|| Arc::from("event dropped without being triggered"));

// ── Backend trait: routing customization point ──────────────────────

/// Routing layer for event completion operations.
///
/// Only three methods — just the operations that need local-vs-remote routing
/// in a distributed setup. [`EventSystemBase`](crate::EventSystemBase) implements
/// this for the local path; a distributed backend would add network routing.
pub trait EventBackend: Send + Sync {
    /// Mark the event as successfully completed, waking all waiters.
    fn trigger(&self, handle: EventHandle) -> Result<()>;

    /// Poison the event with the given reason, waking all waiters with an error.
    fn poison(&self, handle: EventHandle, reason: Arc<str>) -> Result<()>;

    /// Create a future that resolves when the event completes.
    fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter>;
}

// ── Concrete Event ─────────────────────────────────────────────────

/// A single event that can be triggered or poisoned exactly once.
///
/// `Event` is an RAII guard: dropping it without calling [`trigger`](Event::trigger)
/// or [`poison`](Event::poison) automatically poisons the event so waiters are
/// never silently abandoned. To opt out of drop-poisoning (e.g. when handing
/// ownership to a manager-level operation), call [`into_handle`](Event::into_handle).
///
/// `trigger` and `poison` consume `self`, preventing double-completion at
/// compile time.
pub struct Event {
    inner: Option<EventInner>,
}

struct EventInner {
    handle: EventHandle,
    backend: Arc<dyn EventBackend>,
}

impl Event {
    /// Create a new event RAII guard.
    pub(crate) fn new(handle: EventHandle, backend: Arc<dyn EventBackend>) -> Self {
        Self {
            inner: Some(EventInner { handle, backend }),
        }
    }

    /// Take the inner state, disarming the drop guard.
    fn take_inner(&mut self) -> EventInner {
        self.inner.take().expect("event already consumed")
    }

    /// Return the handle that identifies this event.
    pub fn handle(&self) -> EventHandle {
        self.inner.as_ref().expect("event already consumed").handle
    }

    /// Mark the event as successfully completed, waking all waiters.
    /// Consumes the event, disarming the drop guard.
    pub fn trigger(mut self) -> Result<()> {
        let inner = self.take_inner();
        inner.backend.trigger(inner.handle)
    }

    /// Poison the event with the given reason, waking all waiters with an error.
    /// Consumes the event, disarming the drop guard.
    pub fn poison(mut self, reason: impl Into<Arc<str>>) -> Result<()> {
        let inner = self.take_inner();
        inner.backend.poison(inner.handle, reason.into())
    }

    /// Create a future that resolves when this event completes.
    pub fn awaiter(&self) -> Result<EventAwaiter> {
        let inner = self.inner.as_ref().expect("event already consumed");
        inner.backend.awaiter(inner.handle)
    }

    /// Disarm the drop guard and return the bare handle.
    ///
    /// After this call the event will **not** be auto-poisoned on drop.
    /// Use the returned handle with [`EventManager`](crate::EventManager)
    /// methods to complete the event manually.
    pub fn into_handle(mut self) -> EventHandle {
        self.take_inner().handle
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = inner
                .backend
                .poison(inner.handle, Arc::clone(&*DROP_POISON_REASON));
        }
    }
}
