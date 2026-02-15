// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core [`Event`] trait.

use anyhow::Result;
use std::sync::Arc;

use crate::guard::EventGuard;
use crate::handle::EventHandle;
use crate::slot_v2::EventAwaiter;

/// A single event that can be triggered or poisoned exactly once.
///
/// Cloning an event produces a second handle to the *same* underlying event;
/// triggering either handle completes the event for all waiters.
pub trait Event: Clone + Send + Sync {
    /// Return the handle that identifies this event.
    fn handle(&self) -> EventHandle;

    /// Mark the event as successfully completed, waking all waiters.
    fn trigger(&self) -> Result<()>;

    /// Poison the event with the given reason, waking all waiters with an error.
    fn poison(&self, reason: impl Into<Arc<str>>) -> Result<()>;

    /// Create a future that resolves when this event completes.
    fn awaiter(&self) -> Result<EventAwaiter>;

    /// Wrap this event in an [`EventGuard`] that poisons on drop with a
    /// default reason.
    fn into_guard(self) -> EventGuard<Self>
    where
        Self: Sized,
    {
        EventGuard::new(self, "event dropped without being triggered")
    }

    /// Wrap this event in an [`EventGuard`] that poisons on drop with a
    /// custom reason.
    fn into_guard_with_reason(self, reason: impl Into<Arc<str>>) -> EventGuard<Self>
    where
        Self: Sized,
    {
        EventGuard::new(self, reason)
    }
}
