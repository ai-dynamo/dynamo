// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory for creating distributed event systems with a worker identity.

use std::{num::NonZero, sync::Arc};

use crate::local::LocalEventSystem;

/// Factory that creates a [`LocalEventSystem`] pre-configured with a worker_id.
///
/// Use this when events need globally-unique handles that embed a non-zero
/// worker identifier (e.g. in a Nova-managed distributed system).
///
/// For purely local use, call [`LocalEventSystem::new()`] directly instead.
pub struct DistributedEventFactory {
    worker_id: u64,
    system: Arc<LocalEventSystem>,
}

impl DistributedEventFactory {
    /// Create a new factory (and its backing event system) for the given worker.
    pub fn new(worker_id: NonZero<u64>) -> Self {
        Self {
            worker_id: worker_id.get(),
            system: LocalEventSystem::with_worker_id(worker_id.get()),
        }
    }

    /// The worker identity stamped into every handle produced by this factory.
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Borrow the underlying event system.
    pub fn system(&self) -> &Arc<LocalEventSystem> {
        &self.system
    }

    /// Clone the `Arc` to the underlying event system.
    pub fn event_manager(&self) -> Arc<LocalEventSystem> {
        Arc::clone(&self.system)
    }
}
