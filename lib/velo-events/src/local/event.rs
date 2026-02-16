// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use std::sync::{Arc, LazyLock};

use super::system::LocalEventSystem;
use crate::event::Event;
use crate::handle::EventHandle;
use crate::slot::{EventAwaiter, EventEntry};
use crate::status::EventPoison;

/// Static poison reason reused across all drop-triggered poisons.
static DROP_POISON_REASON: LazyLock<Arc<str>> =
    LazyLock::new(|| Arc::from("event dropped without being triggered"));

/// Local event handle with RAII semantics — auto-poisons on drop unless
/// explicitly triggered, poisoned, or disarmed via [`into_handle`](Event::into_handle).
pub struct LocalEvent {
    inner: Option<LocalEventInner>,
}

struct LocalEventInner {
    system: Arc<LocalEventSystem>,
    entry: Arc<EventEntry>,
    handle: EventHandle,
}

impl LocalEvent {
    pub(crate) fn new(
        system: Arc<LocalEventSystem>,
        entry: Arc<EventEntry>,
        handle: EventHandle,
    ) -> Self {
        Self {
            inner: Some(LocalEventInner {
                system,
                entry,
                handle,
            }),
        }
    }

    /// Take the inner state, disarming the drop guard.
    fn take_inner(&mut self) -> LocalEventInner {
        self.inner.take().expect("event already consumed")
    }
}

impl Event for LocalEvent {
    fn handle(&self) -> EventHandle {
        self.inner.as_ref().expect("event already consumed").handle
    }

    fn trigger(mut self) -> Result<()> {
        let inner = self.take_inner();
        inner.system.trigger_local_entry(inner.entry, inner.handle)
    }

    fn poison(mut self, reason: impl Into<Arc<str>>) -> Result<()> {
        let inner = self.take_inner();
        let reason: Arc<str> = reason.into();
        inner.system.poison_local_entry(
            inner.entry,
            inner.handle,
            Arc::new(EventPoison::new(inner.handle, reason)),
        )
    }

    fn awaiter(&self) -> Result<EventAwaiter> {
        let inner = self.inner.as_ref().expect("event already consumed");
        inner.system.awaiter_inner(inner.handle)
    }

    fn into_handle(mut self) -> EventHandle {
        self.take_inner().handle
    }
}

impl Drop for LocalEvent {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let poison = Arc::new(EventPoison::new(
                inner.handle,
                Arc::clone(&*DROP_POISON_REASON),
            ));
            let _ = inner
                .system
                .poison_local_entry(inner.entry, inner.handle, poison);
        }
    }
}
