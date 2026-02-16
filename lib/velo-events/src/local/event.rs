// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::system::LocalEventSystem;
use crate::event::Event;
use crate::handle::EventHandle;
use crate::slot::{EventAwaiter, EventEntry};
use crate::status::EventPoison;

/// Local event handle with ability to trigger exactly once.
#[derive(Clone)]
pub struct LocalEvent {
    inner: Arc<LocalEventInner>,
}

struct LocalEventInner {
    system: Arc<LocalEventSystem>,
    entry: Arc<EventEntry>,
    handle: EventHandle,
    triggered: AtomicBool,
}

impl LocalEvent {
    pub(crate) fn new(
        system: Arc<LocalEventSystem>,
        entry: Arc<EventEntry>,
        handle: EventHandle,
    ) -> Self {
        Self {
            inner: Arc::new(LocalEventInner {
                system,
                entry,
                handle,
                triggered: AtomicBool::new(false),
            }),
        }
    }
}

impl Event for LocalEvent {
    fn handle(&self) -> EventHandle {
        self.inner.handle
    }

    fn trigger(&self) -> Result<()> {
        if self
            .inner
            .triggered
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            bail!("Event {} already triggered", self.inner.handle);
        }

        self.inner
            .system
            .trigger_local_entry(self.inner.entry.clone(), self.inner.handle)
    }

    fn poison(&self, reason: impl Into<Arc<str>>) -> Result<()> {
        let reason: Arc<str> = reason.into();
        self.inner.system.poison_local_entry(
            self.inner.entry.clone(),
            self.inner.handle,
            Arc::new(EventPoison::new(self.inner.handle, reason)),
        )
    }

    fn awaiter(&self) -> Result<EventAwaiter> {
        self.inner.system.awaiter_inner(self.inner.handle)
    }
}
