// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Block Pin Guards
//!
//! This module provides mechanisms to temporarily prevent blocks from being evicted
//! from the inactive pool. This is essential for multi-stage transfers where blocks
//! must remain stable between stages (e.g., D2H followed by H2O).
//!
//! ## Usage
//!
//! ```ignore
//! // Create a registry (typically one per transfer engine)
//! let registry = PinRegistry::new();
//!
//! // After D2H completes, pin the host blocks before triggering H2O
//! let pin_id = Uuid::new_v4();
//! registry.insert(pin_id, PinGuard::new(immutable_blocks));
//!
//! // ... H2O request is sent with pin_id ...
//!
//! // When H2O completes, release the pin
//! registry.remove(&pin_id);  // Blocks can now be evicted
//! ```

use std::any::Any;
use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use super::ImmutableBlocks;
use crate::block_manager::block::{BlockMetadata, locality::LocalityProvider};
use crate::block_manager::storage::Storage;

/// A guard that holds references to blocks, preventing their eviction.
///
/// The guard uses type erasure (`Box<dyn Any>`) to allow storing blocks
/// of different storage types in the same registry.
///
/// When the guard is dropped, the underlying `ImmutableBlock` references
/// are released, allowing the blocks to be returned to the inactive pool
/// and potentially evicted.
pub struct PinGuard {
    /// Type-erased block references. The `Arc` prevents the blocks from
    /// being returned to the inactive pool while this guard exists.
    _blocks: Box<dyn Any + Send + Sync>,
    /// Number of blocks pinned (for debugging/metrics)
    count: usize,
}

impl PinGuard {
    /// Create a new pin guard holding the given immutable blocks.
    ///
    /// The blocks will not be evicted until this guard is dropped.
    pub fn new<S, L, M>(blocks: ImmutableBlocks<S, L, M>) -> Self
    where
        S: Storage + 'static,
        L: LocalityProvider + 'static,
        M: BlockMetadata + 'static,
    {
        let count = blocks.len();
        Self {
            _blocks: Box::new(blocks),
            count,
        }
    }

    /// Create an empty pin guard (for cases where pinning is not needed).
    pub fn empty() -> Self {
        Self {
            _blocks: Box::new(()),
            count: 0,
        }
    }

    /// Returns the number of blocks pinned by this guard.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns true if this guard is empty (pins no blocks).
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl std::fmt::Debug for PinGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinGuard")
            .field("count", &self.count)
            .finish()
    }
}

/// Registry for managing active pin guards.
///
/// This provides a thread-safe way to store and retrieve pin guards
/// by their operation ID. Used by the transfer engine to coordinate
/// between D2H completion and H2O execution.
#[derive(Default, Clone)]
pub struct PinRegistry {
    guards: Arc<DashMap<Uuid, PinGuard>>,
}

impl PinRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a pin guard with the given ID.
    ///
    /// If a guard with this ID already exists, it is replaced and the
    /// old guard is dropped (releasing its pins).
    pub fn insert(&self, id: Uuid, guard: PinGuard) {
        let count = guard.count();
        if let Some(old) = self.guards.insert(id, guard) {
            tracing::warn!(
                pin_id = %id,
                old_count = old.count(),
                new_count = count,
                "Replaced existing pin guard"
            );
        } else {
            tracing::debug!(
                pin_id = %id,
                count = count,
                "Registered pin guard"
            );
        }
    }

    /// Remove and return the pin guard with the given ID.
    ///
    /// Returns `None` if no guard exists with this ID.
    pub fn remove(&self, id: &Uuid) -> Option<PinGuard> {
        let result = self.guards.remove(id).map(|(_, guard)| guard);
        if let Some(ref guard) = result {
            tracing::debug!(
                pin_id = %id,
                count = guard.count(),
                "Released pin guard"
            );
        }
        result
    }

    /// Check if a pin guard exists with the given ID.
    pub fn contains(&self, id: &Uuid) -> bool {
        self.guards.contains_key(id)
    }

    /// Returns the number of active pin guards.
    pub fn len(&self) -> usize {
        self.guards.len()
    }

    /// Returns true if there are no active pin guards.
    pub fn is_empty(&self) -> bool {
        self.guards.is_empty()
    }

    /// Remove all pin guards.
    ///
    /// This releases all pinned blocks, allowing them to be evicted.
    pub fn clear(&self) {
        let count = self.guards.len();
        self.guards.clear();
        if count > 0 {
            tracing::info!(count, "Cleared all pin guards");
        }
    }

    /// Get the total number of blocks currently pinned across all guards.
    pub fn total_pinned_blocks(&self) -> usize {
        self.guards.iter().map(|entry| entry.count()).sum()
    }
}

impl std::fmt::Debug for PinRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinRegistry")
            .field("num_guards", &self.guards.len())
            .field("total_pinned", &self.total_pinned_blocks())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_guard_empty() {
        let guard = PinGuard::empty();
        assert!(guard.is_empty());
        assert_eq!(guard.count(), 0);
    }

    #[test]
    fn test_pin_registry_basic() {
        let registry = PinRegistry::new();
        assert!(registry.is_empty());

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        registry.insert(id1, PinGuard::empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains(&id1));
        assert!(!registry.contains(&id2));

        registry.insert(id2, PinGuard::empty());
        assert_eq!(registry.len(), 2);

        let guard = registry.remove(&id1);
        assert!(guard.is_some());
        assert_eq!(registry.len(), 1);
        assert!(!registry.contains(&id1));

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_pin_registry_replace() {
        let registry = PinRegistry::new();
        let id = Uuid::new_v4();

        registry.insert(id, PinGuard::empty());
        registry.insert(id, PinGuard::empty()); // Replace

        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_pin_registry_clone() {
        let registry = PinRegistry::new();
        let id = Uuid::new_v4();

        registry.insert(id, PinGuard::empty());

        let registry2 = registry.clone();
        assert!(registry2.contains(&id));

        // Both registries share the same underlying map
        registry.remove(&id);
        assert!(!registry2.contains(&id));
    }
}
