// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core [`Event`] trait.

use anyhow::Result;
use std::sync::Arc;

use crate::handle::EventHandle;
use crate::slot::EventAwaiter;

/// A single event that can be triggered or poisoned exactly once.
///
/// `Event` is an RAII guard: dropping it without calling [`trigger`](Event::trigger)
/// or [`poison`](Event::poison) automatically poisons the event so waiters are
/// never silently abandoned. To opt out of drop-poisoning (e.g. when handing
/// ownership to a manager-level operation), call [`into_handle`](Event::into_handle).
///
/// `trigger` and `poison` consume `self`, preventing double-completion at
/// compile time.
pub trait Event: Send + Sync {
    /// Return the handle that identifies this event.
    fn handle(&self) -> EventHandle;

    /// Mark the event as successfully completed, waking all waiters.
    /// Consumes the event, disarming the drop guard.
    fn trigger(self) -> Result<()>;

    /// Poison the event with the given reason, waking all waiters with an error.
    /// Consumes the event, disarming the drop guard.
    fn poison(self, reason: impl Into<Arc<str>>) -> Result<()>;

    /// Create a future that resolves when this event completes.
    fn awaiter(&self) -> Result<EventAwaiter>;

    /// Disarm the drop guard and return the bare handle.
    ///
    /// After this call the event will **not** be auto-poisoned on drop.
    /// Use the returned handle with [`EventManager`](crate::EventManager)
    /// methods to complete the event manually.
    fn into_handle(self) -> EventHandle;
}
