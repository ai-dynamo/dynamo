// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete [`EventManager`] that ties lifecycle and routing together.

use anyhow::Result;
use std::sync::Arc;

use crate::base::EventSystemBase;
use crate::event::{Event, EventBackend};
use crate::handle::EventHandle;
use crate::slot::EventAwaiter;
use crate::status::EventStatus;

/// Manages a collection of events — creating, triggering, poisoning, and
/// merging them.
///
/// `EventManager` is `Clone` and `Send + Sync`, so it can be cheaply shared
/// across async tasks.
///
/// # Local vs distributed
///
/// [`EventManager::local()`] creates a purely local manager backed by
/// [`EventSystemBase`]. For distributed setups, construct a manager with
/// [`EventManager::new()`] providing a custom [`EventBackend`] that routes
/// remote handles over the network.
#[derive(Clone)]
pub struct EventManager {
    base: Arc<EventSystemBase>,
    backend: Arc<dyn EventBackend>,
}

impl EventManager {
    /// Create a purely local event manager.
    ///
    /// The [`EventSystemBase`] is used as both the lifecycle store and
    /// the completion backend.
    pub fn local() -> Self {
        let base = EventSystemBase::local();
        let backend = base.clone() as Arc<dyn EventBackend>;
        Self { base, backend }
    }

    /// Create an event manager with a custom backend for routing.
    ///
    /// Used for distributed setups where trigger/poison/awaiter may be routed
    /// over the network.
    pub fn new(base: Arc<EventSystemBase>, backend: Arc<dyn EventBackend>) -> Self {
        Self { base, backend }
    }

    /// The system identity stamped into every handle produced by this manager.
    pub fn system_id(&self) -> u64 {
        self.base.system_id()
    }

    /// Borrow the underlying event system base.
    pub fn base(&self) -> &Arc<EventSystemBase> {
        &self.base
    }

    /// Allocate a new pending event.
    pub fn new_event(&self) -> Result<Event> {
        self.base.new_event_with_backend(self.backend.clone())
    }

    /// Create a future that resolves when the given event completes.
    pub fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter> {
        self.backend.awaiter(handle)
    }

    /// Non-blocking status check.
    pub fn poll(&self, handle: EventHandle) -> Result<EventStatus> {
        self.base.poll_inner(handle)
    }

    /// Trigger the event identified by `handle`.
    pub fn trigger(&self, handle: EventHandle) -> Result<()> {
        self.backend.trigger(handle)
    }

    /// Poison the event identified by `handle` with the given reason.
    pub fn poison(&self, handle: EventHandle, reason: impl Into<Arc<str>>) -> Result<()> {
        self.backend.poison(handle, reason.into())
    }

    /// Create a new event that completes when **all** `inputs` complete.
    ///
    /// If any input is poisoned the merged event is poisoned with the
    /// accumulated reasons.
    pub fn merge_events(&self, inputs: Vec<EventHandle>) -> Result<EventHandle> {
        self.base
            .merge_events_with(inputs, self.backend.clone())
    }

    /// Poison every pending event and reject future allocations.
    pub fn force_shutdown(&self, reason: impl Into<Arc<str>>) {
        self.base.force_shutdown_inner(reason)
    }

    /// Return the poison reason for a completed generation, if any.
    #[allow(dead_code)]
    pub(crate) fn poison_reason(&self, handle: EventHandle) -> Option<Arc<str>> {
        self.base.poison_reason(handle)
    }
}
