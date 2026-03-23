// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event status types shared across local and distributed implementations.

use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use crate::handle::EventHandle;

/// Alias for event generation counters.
pub type Generation = u32;

/// Status returned from non-blocking event queries.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum EventStatus {
    Pending,
    Ready,
    Poisoned,
}

/// Describes a poisoned event generation.
#[derive(Clone, Debug)]
pub struct EventPoison {
    handle: EventHandle,
    reason: Arc<str>,
}

impl EventPoison {
    /// Create a new poisoned event.
    pub fn new(handle: EventHandle, reason: impl Into<Arc<str>>) -> Self {
        Self {
            handle,
            reason: reason.into(),
        }
    }

    /// Get the handle of the poisoned event.
    pub fn handle(&self) -> EventHandle {
        self.handle
    }

    /// Get the reason of the poisoned event.
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Get the reason of the poisoned event as an `Arc<str>`.
    pub fn reason_arc(&self) -> &Arc<str> {
        &self.reason
    }
}

impl Display for EventPoison {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Event {} poisoned: {}", self.handle, self.reason())
    }
}

impl std::error::Error for EventPoison {}
