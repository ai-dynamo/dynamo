// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Pool Registry
//!
//! Provides a dynamic registry for block pools, replacing the fixed three-pool
//! structure (device / host / disk) with a [`PoolRegistry`] that supports:
//!
//! - Dynamic registration of pools at any [`CacheLevel`]
//! - Multiple pools per cache level (e.g. multiple device pools)
//! - Backward-compatible accessors for legacy G1/G2/G3 code paths
//!
//! This is Phase 1 of the V2 migration path: the registry introduces the
//! indirection layer that future phases will use to integrate V2's
//! `TransportManager` and `PhysicalLayout`.

use super::block::{BlockMetadata, locality::LocalityProvider};
use super::offload::filter::OffloadFilter;
use super::pool::BlockPool;
use super::storage::{DeviceStorage, DiskStorage, PinnedStorage};
use super::CacheLevel;
use std::collections::HashMap;
use std::sync::Arc;

pub type PoolId = u32;

/// A type-safe wrapper around a concrete pool with its storage kind.
///
/// Because [`BlockPool`] is parameterized by the storage type, we use an enum
/// to hold pools of different storage types in a single collection.
pub enum PoolKind<L: LocalityProvider, M: BlockMetadata> {
    Device(Arc<dyn BlockPool<DeviceStorage, L, M>>),
    Host(Arc<dyn BlockPool<PinnedStorage, L, M>>),
    Disk(Arc<dyn BlockPool<DiskStorage, L, M>>),
}

impl<L: LocalityProvider, M: BlockMetadata> PoolKind<L, M> {
    pub fn storage_name(&self) -> &'static str {
        match self {
            PoolKind::Device(_) => "device",
            PoolKind::Host(_) => "host",
            PoolKind::Disk(_) => "disk",
        }
    }

    pub fn as_device(&self) -> Option<&dyn BlockPool<DeviceStorage, L, M>> {
        match self {
            PoolKind::Device(p) => Some(p.as_ref()),
            _ => None,
        }
    }

    pub fn as_host(&self) -> Option<&dyn BlockPool<PinnedStorage, L, M>> {
        match self {
            PoolKind::Host(p) => Some(p.as_ref()),
            _ => None,
        }
    }

    pub fn as_disk(&self) -> Option<&dyn BlockPool<DiskStorage, L, M>> {
        match self {
            PoolKind::Disk(p) => Some(p.as_ref()),
            _ => None,
        }
    }

    pub fn as_device_arc(&self) -> Option<Arc<dyn BlockPool<DeviceStorage, L, M>>> {
        match self {
            PoolKind::Device(p) => Some(p.clone()),
            _ => None,
        }
    }

    pub fn as_host_arc(&self) -> Option<Arc<dyn BlockPool<PinnedStorage, L, M>>> {
        match self {
            PoolKind::Host(p) => Some(p.clone()),
            _ => None,
        }
    }

    pub fn as_disk_arc(&self) -> Option<Arc<dyn BlockPool<DiskStorage, L, M>>> {
        match self {
            PoolKind::Disk(p) => Some(p.clone()),
            _ => None,
        }
    }
}

/// Metadata for a registered pool.
pub struct PoolEntry<L: LocalityProvider, M: BlockMetadata> {
    pub id: PoolId,
    pub cache_level: CacheLevel,
    pub kind: PoolKind<L, M>,
    pub offload_filter: Option<Arc<dyn OffloadFilter>>,
}

/// A dynamic registry of block pools keyed by [`PoolId`] and indexed by
/// [`CacheLevel`].
///
/// Replaces the fixed `disk_pool` / `host_pool` / `device_pool` fields in
/// `KvBlockManagerState` with a structure that supports heterogeneous and
/// multiple pools per tier.
pub struct PoolRegistry<L: LocalityProvider, M: BlockMetadata> {
    pools: HashMap<PoolId, PoolEntry<L, M>>,
    by_level: HashMap<CacheLevel, Vec<PoolId>>,
    next_id: PoolId,
}

impl<L: LocalityProvider, M: BlockMetadata> PoolRegistry<L, M> {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            by_level: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a pool at the given cache level. Returns the assigned [`PoolId`].
    pub fn register(
        &mut self,
        cache_level: CacheLevel,
        kind: PoolKind<L, M>,
        offload_filter: Option<Arc<dyn OffloadFilter>>,
    ) -> PoolId {
        let id = self.next_id;
        self.next_id += 1;

        tracing::debug!(
            "PoolRegistry: registered {} pool at {:?} with id={}",
            kind.storage_name(),
            cache_level,
            id
        );

        let entry = PoolEntry {
            id,
            cache_level,
            kind,
            offload_filter,
        };

        self.pools.insert(id, entry);
        self.by_level.entry(cache_level).or_default().push(id);
        id
    }

    /// Returns all pool entries registered at the given cache level.
    pub fn pools_at_level(&self, level: &CacheLevel) -> Vec<&PoolEntry<L, M>> {
        self.by_level
            .get(level)
            .map(|ids| ids.iter().filter_map(|id| self.pools.get(id)).collect())
            .unwrap_or_default()
    }

    /// Returns a specific pool entry by ID.
    pub fn get(&self, id: &PoolId) -> Option<&PoolEntry<L, M>> {
        self.pools.get(id)
    }

    /// Returns all registered pool IDs.
    pub fn pool_ids(&self) -> Vec<PoolId> {
        self.pools.keys().copied().collect()
    }

    /// Returns the number of registered pools.
    pub fn len(&self) -> usize {
        self.pools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    // ── Backward-compatible typed accessors ────────────────────────────
    // Return the FIRST pool of each concrete type, matching the semantics
    // of the old fixed-field approach in `KvBlockManagerState`.

    /// Returns the first registered device (G1) pool, if any.
    pub fn device(&self) -> Option<&dyn BlockPool<DeviceStorage, L, M>> {
        self.pools_at_level(&CacheLevel::G1)
            .into_iter()
            .find_map(|e| e.kind.as_device())
    }

    /// Returns the first registered host (G2) pool, if any.
    pub fn host(&self) -> Option<&dyn BlockPool<PinnedStorage, L, M>> {
        self.pools_at_level(&CacheLevel::G2)
            .into_iter()
            .find_map(|e| e.kind.as_host())
    }

    /// Returns the first registered disk (G3) pool, if any.
    pub fn disk(&self) -> Option<&dyn BlockPool<DiskStorage, L, M>> {
        self.pools_at_level(&CacheLevel::G3)
            .into_iter()
            .find_map(|e| e.kind.as_disk())
    }

    /// Returns the first registered device pool as `Arc`, if any.
    pub fn device_arc(&self) -> Option<Arc<dyn BlockPool<DeviceStorage, L, M>>> {
        self.pools_at_level(&CacheLevel::G1)
            .into_iter()
            .find_map(|e| e.kind.as_device_arc())
    }

    /// Returns the first registered host pool as `Arc`, if any.
    pub fn host_arc(&self) -> Option<Arc<dyn BlockPool<PinnedStorage, L, M>>> {
        self.pools_at_level(&CacheLevel::G2)
            .into_iter()
            .find_map(|e| e.kind.as_host_arc())
    }

    /// Returns the first registered disk pool as `Arc`, if any.
    pub fn disk_arc(&self) -> Option<Arc<dyn BlockPool<DiskStorage, L, M>>> {
        self.pools_at_level(&CacheLevel::G3)
            .into_iter()
            .find_map(|e| e.kind.as_disk_arc())
    }

    /// Collect offload filters for the three standard tiers.
    /// Used during OffloadManager construction.
    pub fn offload_filters(
        &self,
    ) -> (
        Option<Arc<dyn OffloadFilter>>,
        Option<Arc<dyn OffloadFilter>>,
        Option<Arc<dyn OffloadFilter>>,
    ) {
        let device_filter = self
            .pools_at_level(&CacheLevel::G1)
            .into_iter()
            .find_map(|e| e.offload_filter.clone());
        let host_filter = self
            .pools_at_level(&CacheLevel::G2)
            .into_iter()
            .find_map(|e| e.offload_filter.clone());
        let disk_filter = self
            .pools_at_level(&CacheLevel::G3)
            .into_iter()
            .find_map(|e| e.offload_filter.clone());
        (device_filter, host_filter, disk_filter)
    }
}

impl<L: LocalityProvider, M: BlockMetadata> Default for PoolRegistry<L, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: LocalityProvider, M: BlockMetadata> std::fmt::Debug for PoolRegistry<L, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let levels: Vec<_> = self
            .by_level
            .iter()
            .map(|(level, ids)| format!("{:?}({})", level, ids.len()))
            .collect();
        f.debug_struct("PoolRegistry")
            .field("num_pools", &self.pools.len())
            .field("levels", &levels)
            .finish()
    }
}
