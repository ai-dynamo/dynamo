// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed event backend implementing `velo_events::EventBackend`.

use anyhow::Result;
use std::sync::Arc;

use velo_events::{EventAwaiter, EventBackend, EventHandle};

use super::VeloEvents;

/// Routes event operations to local or remote based on handle ownership.
pub(crate) struct DistributedEventBackend {
    pub(crate) events: Arc<VeloEvents>,
}

impl EventBackend for DistributedEventBackend {
    fn trigger(&self, handle: EventHandle) -> Result<()> {
        if handle.system_id() == self.events.system_id {
            self.events.local_base.trigger_inner(handle)
        } else {
            self.events.trigger_remote_fire_and_forget(handle)
        }
    }

    fn poison(&self, handle: EventHandle, reason: Arc<str>) -> Result<()> {
        if handle.system_id() == self.events.system_id {
            self.events.local_base.poison_inner(handle, reason)
        } else {
            self.events.poison_remote_fire_and_forget(handle, reason)
        }
    }

    fn awaiter(&self, handle: EventHandle) -> Result<EventAwaiter> {
        if handle.system_id() == self.events.system_id {
            self.events.local_base.awaiter_inner(handle)
        } else {
            self.events.wait_remote(handle)
        }
    }
}
