// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`EventManager`] trait for creating and managing events.

use anyhow::Result;
use std::sync::Arc;

use crate::event::Event;
use crate::handle::EventHandle;
use crate::slot::EventAwaiter;
use crate::status::EventStatus;

/// Manages a collection of events — creating, triggering, poisoning, and
/// merging them.
///
/// Implementations are expected to be `Send + Sync` so they can be shared
/// across async tasks.
pub trait EventManager: Send + Sync {
    /// The concrete event type produced by this manager.
    type Event: Event;

    /// Allocate a new pending event.
    fn new_event(&self) -> Result<Self::Event>;

    /// Create a future that resolves when the given event completes.
    fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter>;

    /// Non-blocking status check.
    fn poll(&self, handle: EventHandle) -> Result<EventStatus>;

    /// Trigger the event identified by `handle`.
    fn trigger(&self, handle: EventHandle) -> Result<()>;

    /// Poison the event identified by `handle` with the given reason.
    fn poison(&self, handle: EventHandle, reason: impl Into<Arc<str>>) -> Result<()>;

    /// Create a new event that completes when **all** `inputs` complete.
    ///
    /// If any input is poisoned the merged event is poisoned with the
    /// accumulated reasons.
    fn merge_events(&self, inputs: Vec<EventHandle>) -> Result<EventHandle>;

    /// Poison every pending event and reject future allocations.
    fn force_shutdown(&self, reason: impl Into<Arc<str>>);
}
