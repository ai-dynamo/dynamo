// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory for creating distributed event systems with a system identity.

use std::{num::NonZero, sync::Arc};

use crate::base::EventSystemBase;
use crate::manager::EventManager;

/// Factory that creates an [`EventManager`] pre-configured with a system_id.
///
/// Use this when events need globally-unique handles that embed a non-zero
/// system identifier (e.g. in a Nova-managed distributed system).
///
/// For purely local use, call [`EventManager::local()`] directly instead.
pub struct DistributedEventFactory {
    system_id: u64,
    base: Arc<EventSystemBase>,
}

impl DistributedEventFactory {
    /// Create a new factory (and its backing event system) for the given system.
    pub fn new(system_id: NonZero<u64>) -> Self {
        Self {
            system_id: system_id.get(),
            base: EventSystemBase::distributed(system_id.get()),
        }
    }

    /// The system identity stamped into every handle produced by this factory.
    pub fn system_id(&self) -> u64 {
        self.system_id
    }

    /// Borrow the underlying event system base.
    pub fn system(&self) -> &Arc<EventSystemBase> {
        &self.base
    }

    /// Create an [`EventManager`] backed by this factory's system.
    ///
    /// Currently uses the local backend; a future distributed backend will
    /// route remote handles over the network.
    pub fn event_manager(&self) -> EventManager {
        EventManager::new(self.base.clone(), self.base.clone() as _)
    }
}
