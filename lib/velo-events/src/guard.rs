// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard that poisons an event on drop unless explicitly triggered.

use anyhow::Result;
use std::sync::Arc;

use crate::event::Event;
use crate::handle::EventHandle;
use crate::slot_v2::EventAwaiter;

/// RAII wrapper around an [`Event`] that automatically poisons the event if
/// dropped without being triggered.
pub struct EventGuard<E: Event> {
    event: Option<E>,
    poison_reason: Arc<str>,
}

impl<E: Event> EventGuard<E> {
    /// Create a new guard that will poison the event with `poison_reason`
    /// if dropped without an explicit call to [`trigger`](Self::trigger) or
    /// [`poison`](Self::poison).
    pub fn new(event: E, poison_reason: impl Into<Arc<str>>) -> Self {
        Self {
            event: Some(event),
            poison_reason: poison_reason.into(),
        }
    }

    /// Return the handle of the guarded event.
    pub fn handle(&self) -> EventHandle {
        self.event
            .as_ref()
            .expect("guard already consumed")
            .handle()
    }

    /// Trigger the guarded event, consuming the guard.
    pub fn trigger(mut self) -> Result<()> {
        self.event.take().expect("guard already consumed").trigger()
    }

    /// Poison the guarded event with a custom reason, consuming the guard.
    pub fn poison(mut self, reason: impl Into<Arc<str>>) -> Result<()> {
        self.event
            .take()
            .expect("guard already consumed")
            .poison(reason)
    }

    /// Create a future that resolves when the guarded event completes.
    pub fn awaiter(&self) -> Result<EventAwaiter> {
        self.event
            .as_ref()
            .expect("guard already consumed")
            .awaiter()
    }
}

impl<E: Event> Drop for EventGuard<E> {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            let _ = event.poison(self.poison_reason.clone());
        }
    }
}
