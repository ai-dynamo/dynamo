// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-group residency for hybrid (attention + recurrent state) models.
//!
//! vLLM publishes one `BlockStored` per KV cache group at the same positional
//! block hash. For the main-attention group that hash names a reusable block;
//! for a recurrent (Mamba / GDN / linear-attention) group it names the state
//! checkpoint at that boundary. A hybrid engine can resume a prefix only at a
//! boundary where every group has a residency, so an attention-only match
//! overestimates the reusable prefix and over-sizes KV transfer hints
//! (ai-dynamo/dynamo#16794).
//!
//! Recurrent events update per-owner checkpoint membership, never attention edges.
//! Complete layouts include groups with no resident checkpoints and survive dumps.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use arc_swap::{ArcSwap, Guard};
use dashmap::{DashMap, mapref::entry::Entry};
use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;
type OwnerAlignments = FxHashMap<ResidencyOwner, usize>;

/// One current checkpoint record. Group completeness is resolved on the write
/// side; readers need neither owner-layout joins nor publication certificates.
#[derive(Debug, Default)]
struct CheckpointPresence {
    owners: [OwnerAlignments; 4],
    device_workers: Arc<FxHashSet<WorkerWithDpRank>>,
}

impl CheckpointPresence {
    fn update(
        &mut self,
        owner: ResidencyOwner,
        mask: u8,
        alignment: usize,
        shared_workers: &mut Option<Arc<FxHashSet<WorkerWithDpRank>>>,
    ) {
        for (index, owners) in self.owners.iter_mut().enumerate() {
            if mask & (1 << index) != 0 {
                owners.insert(owner, alignment);
            } else {
                owners.remove(&owner);
            }
        }
        if let Some(worker) = owner.as_worker() {
            let include = mask & 1 != 0 && alignment == 1;
            if self.device_workers.contains(&worker) != include {
                let workers = Arc::make_mut(&mut self.device_workers);
                if include {
                    workers.insert(worker);
                } else {
                    workers.remove(&worker);
                }
                // Reuse an equal set only inside this event's update batch.
                // There is no persistent cohort cache or validity object.
                if let Some(shared) = shared_workers.as_ref()
                    && shared.as_ref() == self.device_workers.as_ref()
                {
                    self.device_workers = shared.clone();
                } else {
                    *shared_workers = Some(self.device_workers.clone());
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.owners.iter().all(FxHashMap::is_empty)
    }
}

#[derive(Debug, Default)]
struct CheckpointDirectory {
    entries: DashMap<u64, Arc<RwLock<CheckpointPresence>>, FxBuildHasher>,
}

impl CheckpointDirectory {
    fn record(&self, hash: ExternalSequenceBlockHash) -> Option<Arc<RwLock<CheckpointPresence>>> {
        // Drop the directory's shard guard before waiting for the record lock.
        self.entries
            .get(&hash.0)
            .map(|record| record.value().clone())
    }

    fn publish(&self, owner: ResidencyOwner, updates: Vec<(u64, u8)>, alignment: usize) {
        let mut shared_workers = None;
        for (hash, mask) in updates {
            // Writers keep one directory entry guard through a single record
            // update. This serializes removal with other writers and prevents
            // writes into a detached record. No reader holds a directory guard
            // while taking a record lock, and no record reader acquires an owner
            // lock. Empty records are reclaimed immediately.
            match self.entries.entry(hash) {
                Entry::Occupied(entry) => {
                    let empty = {
                        let mut record = entry.get().write();
                        record.update(owner, mask, alignment, &mut shared_workers);
                        record.is_empty()
                    };
                    if empty {
                        entry.remove();
                    }
                }
                Entry::Vacant(entry) if mask != 0 => {
                    let mut record = CheckpointPresence::default();
                    record.update(owner, mask, alignment, &mut shared_workers);
                    entry.insert(Arc::new(RwLock::new(record)));
                }
                Entry::Vacant(_) => {}
            }
        }
    }

    fn with_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&OwnerAlignments),
    ) -> bool {
        let Some(record) = self.record(hash) else {
            return false;
        };
        let record = record.read();
        let owners = &record.owners[tier_index(tier)];
        if owners.is_empty() {
            return false;
        }
        visit(owners);
        true
    }
}

fn compare_owners(left: &ResidencyOwner, right: &ResidencyOwner) -> std::cmp::Ordering {
    match (left, right) {
        (ResidencyOwner::Worker(left), ResidencyOwner::Worker(right)) => left.cmp(right),
        (ResidencyOwner::CacheOwner(left), ResidencyOwner::CacheOwner(right)) => left.cmp(right),
        (ResidencyOwner::Worker(_), ResidencyOwner::CacheOwner(_)) => std::cmp::Ordering::Less,
        (ResidencyOwner::CacheOwner(_), ResidencyOwner::Worker(_)) => std::cmp::Ordering::Greater,
    }
}

fn tier_index(tier: StorageTier) -> usize {
    match tier {
        StorageTier::Device => 0,
        StorageTier::HostPinned => 1,
        StorageTier::Disk => 2,
        StorageTier::External => 3,
    }
}

fn tier_bit(tier: StorageTier) -> u8 {
    1 << tier_index(tier)
}

#[derive(Debug, Default)]
struct OwnerPresence {
    groups: GroupPresence,
    group_counts: FxHashMap<u64, [u32; 4]>,
    complete_checkpoints: FxHashMap<u64, u8>,
    declared: bool,
    expected_group_count: Option<usize>,
    alignment_blocks: usize,
    retired: bool,
    published_general_device_eval: bool,
}

impl OwnerPresence {
    fn refresh_general_requirement(&mut self, count: &AtomicU64) {
        let requires_general = !self.retired && self.alignment_blocks > 1;
        if requires_general != self.published_general_device_eval {
            if requires_general {
                count.fetch_add(1, Ordering::Release);
            } else {
                count.fetch_sub(1, Ordering::Release);
            }
            self.published_general_device_eval = requires_general;
        }
    }

    fn layout(&self) -> CheckpointLayout {
        CheckpointLayout {
            alignment_blocks: self.alignment_blocks.max(1),
            complete_layout: !self.retired
                && !self.groups.is_empty()
                && self.expected_group_count == Some(self.groups.len()),
        }
    }

    fn refresh_checkpoint(&mut self, hash: u64) {
        let mask = if self.layout().complete_layout {
            self.group_counts
                .get(&hash)
                .map_or(0, |counts| complete_tiers(counts, self.groups.len()))
        } else {
            0
        };
        if mask == 0 {
            self.complete_checkpoints.remove(&hash);
        } else {
            self.complete_checkpoints.insert(hash, mask);
        }
    }

    fn rebuild_complete_checkpoints(&mut self) {
        self.complete_checkpoints.clear();
        if self.layout().complete_layout {
            for (&hash, counts) in &self.group_counts {
                let mask = complete_tiers(counts, self.groups.len());
                if mask != 0 {
                    self.complete_checkpoints.insert(hash, mask);
                }
            }
        }
    }

    fn rebuild_group_counts(&mut self) {
        self.group_counts.clear();
        for group in self.groups.values() {
            for (&hash, &mask) in group {
                let counts = self.group_counts.entry(hash).or_default();
                for (index, count) in counts.iter_mut().enumerate() {
                    *count += u32::from(mask & (1 << index) != 0);
                }
            }
        }
        self.rebuild_complete_checkpoints();
    }

    fn store(&mut self, group_idx: u32, hash: u64, tier: StorageTier) {
        let bit = tier_bit(tier);
        let mask = self
            .groups
            .entry(group_idx)
            .or_default()
            .entry(hash)
            .or_default();
        if *mask & bit != 0 {
            return;
        }
        *mask |= bit;
        self.group_counts.entry(hash).or_default()[tier_index(tier)] += 1;
        self.refresh_checkpoint(hash);
    }

    fn remove(&mut self, group_idx: u32, hash: u64, tier: StorageTier) {
        let Some(group) = self.groups.get_mut(&group_idx) else {
            return;
        };
        let Some(mask) = group.get_mut(&hash) else {
            return;
        };
        let bit = tier_bit(tier);
        if *mask & bit == 0 {
            return;
        }
        *mask &= !bit;
        if *mask == 0 {
            group.remove(&hash);
        }
        if let Some(counts) = self.group_counts.get_mut(&hash) {
            counts[tier_index(tier)] -= 1;
            if counts.iter().all(|&count| count == 0) {
                self.group_counts.remove(&hash);
            }
        }
        self.refresh_checkpoint(hash);
    }

    fn clear_checkpoints(&mut self) {
        for group in self.groups.values_mut() {
            group.clear();
        }
        self.group_counts.clear();
        self.complete_checkpoints.clear();
    }
}

fn complete_tiers(counts: &[u32; 4], required_groups: usize) -> u8 {
    counts.iter().enumerate().fold(0, |mask, (index, &count)| {
        mask | (u8::from(count as usize == required_groups) << index)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLayout {
    pub alignment_blocks: usize,
    pub complete_layout: bool,
}

/// Only registration topology is shared as a snapshot. Checkpoint data lives in
/// independent mutable records and has the same best-effort freshness as KV facts.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, Arc<RwLock<OwnerPresence>>>,
    cache_owners: FxHashMap<ResidencyOwnerKey, ResidencyOwner>,
}

#[derive(Debug)]
#[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
pub(crate) struct KnownWorkerRead {
    registry: Guard<Arc<OwnerRegistry>>,
}

#[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
impl KnownWorkerRead {
    pub(crate) fn is_known_worker(&self, worker: WorkerWithDpRank) -> bool {
        self.registry
            .owners
            .contains_key(&ResidencyOwner::worker(worker))
    }
}

/// Owner classification is captured once; checkpoint lookups read current
/// records. A previously unknown owner remains unknown for this query.
#[derive(Debug)]
pub struct CheckpointReadView {
    registry: Arc<OwnerRegistry>,
    layouts: FxHashMap<ResidencyOwner, Option<CheckpointLayout>>,
    checkpoints: Arc<CheckpointDirectory>,
}

impl CheckpointReadView {
    pub fn capture_layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        if let Some(&layout) = self.layouts.get(&owner) {
            return layout;
        }
        let state = self.registry.owners.get(&owner)?.read();
        // A registered owner remains known to this query even if retirement
        // happens before its first layout read. Its retired layout is incomplete,
        // so absent checkpoints receive zero credit rather than dense KV credit.
        Some(state.layout())
    }

    pub fn remember_layout(&mut self, owner: ResidencyOwner, captured: Option<CheckpointLayout>) {
        self.layouts.entry(owner).or_insert(captured);
    }

    pub fn track_owner(&mut self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        let captured = self.capture_layout(owner);
        *self.layouts.entry(owner).or_insert(captured)
    }

    pub fn layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        self.layouts.get(&owner).copied().flatten()
    }

    pub fn owner_for_key(&self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        self.registry.cache_owners.get(&key).copied()
    }

    pub fn track_cache_owner(&mut self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        let owner = self.owner_for_key(key)?;
        self.track_owner(owner);
        Some(owner)
    }

    /// The callback receives current complete owners and their alignment from
    /// one record read. It must not acquire owner locks or recursively enter the
    /// directory while holding this record read.
    pub fn with_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&OwnerAlignments),
    ) -> bool {
        self.checkpoints.with_owners(hash, tier, visit)
    }

    pub fn owner_layouts_at(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> impl Iterator<Item = (ResidencyOwner, CheckpointLayout)> {
        let mut owners = Vec::new();
        self.with_owners(hash, tier, |current| {
            owners.extend(current.iter().map(|(&owner, &alignment_blocks)| {
                (
                    owner,
                    CheckpointLayout {
                        alignment_blocks,
                        complete_layout: true,
                    },
                )
            }));
        });
        owners.sort_by(|(left, _), (right, _)| compare_owners(left, right));
        owners.into_iter()
    }

    /// Compatibility enumeration for explicitly tracked owners. Unlike the old
    /// implementation this does not retain checkpoint or shard snapshots.
    pub fn owners_at(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> impl Iterator<Item = ResidencyOwner> {
        let mut owners = Vec::new();
        self.with_owners(hash, tier, |current| {
            owners.extend(
                current
                    .keys()
                    .copied()
                    .filter(|owner| self.layouts.get(owner).is_some_and(Option::is_some)),
            );
        });
        owners.sort_by(compare_owners);
        owners.into_iter()
    }
}

#[derive(Debug, Default)]
pub struct KvCacheGroupResidency {
    registry: ArcSwap<OwnerRegistry>,
    registry_writer: Mutex<()>,
    active: AtomicBool,
    checkpoints: Arc<CheckpointDirectory>,
    general_device_eval_layouts: AtomicU64,
}

impl KvCacheGroupResidency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn requires_general_device_eval(&self) -> bool {
        self.general_device_eval_layouts.load(Ordering::Acquire) != 0
    }

    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn known_worker_read(&self) -> KnownWorkerRead {
        KnownWorkerRead {
            registry: self.registry.load(),
        }
    }

    /// Visit the current complete, alignment-one Device workers for this hash.
    /// Only one record is read; there is no query-time layout join or validity fence.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn with_checkpoint_workers(
        &self,
        hash: ExternalSequenceBlockHash,
        visit: impl FnOnce(&Arc<FxHashSet<WorkerWithDpRank>>),
    ) -> bool {
        let Some(record) = self.checkpoints.record(hash) else {
            return false;
        };
        let record = record.read();
        if record.device_workers.is_empty() {
            return false;
        }
        visit(&record.device_workers);
        true
    }

    pub fn read_view(&self) -> CheckpointReadView {
        CheckpointReadView {
            registry: self.registry.load_full(),
            layouts: FxHashMap::default(),
            checkpoints: self.checkpoints.clone(),
        }
    }

    fn owner(&self, owner: ResidencyOwner) -> Option<Arc<RwLock<OwnerPresence>>> {
        self.registry.load().owners.get(&owner).cloned()
    }

    fn publish_reset(
        &self,
        owner: ResidencyOwner,
        previous: FxHashMap<u64, u8>,
        state: &OwnerPresence,
    ) {
        let mut updates = state
            .complete_checkpoints
            .iter()
            .map(|(&hash, &mask)| (hash, mask))
            .collect::<Vec<_>>();
        updates.extend(
            previous
                .into_keys()
                .filter(|hash| !state.complete_checkpoints.contains_key(hash))
                .map(|hash| (hash, 0)),
        );
        self.checkpoints
            .publish(owner, updates, state.alignment_blocks.max(1));
    }

    fn clear_owner(&self, owner: ResidencyOwner, state: &mut OwnerPresence) {
        let previous = std::mem::take(&mut state.complete_checkpoints);
        state.clear_checkpoints();
        state.refresh_general_requirement(&self.general_device_eval_layouts);
        self.publish_reset(owner, previous, state);
    }

    fn get_or_create(&self, owner: ResidencyOwner) -> Arc<RwLock<OwnerPresence>> {
        if let Some(state) = self.owner(owner) {
            return state;
        }
        // The same lock serializes retirement and re-registration. Old handles
        // are marked retired before a replacement can publish any checkpoints.
        let _writer = self.registry_writer.lock();
        let current = self.registry.load_full();
        if let Some(state) = current.owners.get(&owner) {
            return state.clone();
        }
        let mut next = (*current).clone();
        let state = Arc::new(RwLock::new(OwnerPresence::default()));
        next.owners.insert(owner, state.clone());
        if matches!(owner, ResidencyOwner::CacheOwner(_)) {
            next.cache_owners.insert(owner.compact_key(), owner);
        }
        self.registry.store(Arc::new(next));
        self.active.store(true, Ordering::Relaxed);
        state
    }

    pub fn declare_groups(&self, owner: ResidencyOwner, groups: &[u32]) {
        self.install_layout(owner, groups, None);
    }

    pub fn declare_layout(&self, owner: ResidencyOwner, groups: &[u32], alignment_blocks: u32) {
        self.install_layout(owner, groups, Some(alignment_blocks.max(1) as usize));
    }

    fn install_layout(&self, owner: ResidencyOwner, groups: &[u32], alignment: Option<usize>) {
        if groups.is_empty() {
            if let ResidencyOwner::Worker(worker) = owner {
                self.remove_worker_rank(worker);
            }
            return;
        }
        let state = self.get_or_create(owner);
        let mut state = state.write();
        if state.retired {
            return;
        }
        let previous_group_count = state.groups.len();
        let previous_expected_count = state.expected_group_count;
        let removed_groups = state.groups.keys().any(|group| !groups.contains(group));
        let layout_changed = previous_group_count != groups.len()
            || !groups.iter().all(|group| state.groups.contains_key(group))
            || previous_expected_count != Some(groups.len())
            || alignment.is_some_and(|alignment| state.alignment_blocks.max(1) != alignment);
        let previous = layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
        let changed = state.declared
            && (state.groups.len() != groups.len()
                || !groups.iter().all(|group| state.groups.contains_key(group))
                || alignment.is_some_and(|alignment| state.alignment_blocks.max(1) != alignment));
        if changed {
            state.groups.clear();
        }
        state.groups.retain(|group, _| groups.contains(group));
        for &group in groups {
            state.groups.entry(group).or_default();
        }
        state.declared = true;
        state.expected_group_count = Some(groups.len());
        if let Some(alignment) = alignment {
            state.alignment_blocks = alignment;
        }
        if changed || removed_groups {
            state.rebuild_group_counts();
        } else if layout_changed {
            state.rebuild_complete_checkpoints();
        }
        if let Some(previous) = previous {
            state.refresh_general_requirement(&self.general_device_eval_layouts);
            self.publish_reset(owner, previous, &state);
        }
    }

    /// Tagged events are consumed even if unsupported; never send them to attention.
    pub fn apply(&self, event: &RouterEvent) -> bool {
        let Some(tag) = event.kv_cache_group else {
            return false;
        };
        if tag.kind != KvCacheGroupKind::Mamba {
            return true;
        }
        let Ok(owner) = event.residency_owner() else {
            return true;
        };
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                let state = self.get_or_create(owner);
                let mut state = state.write();
                if state.retired {
                    return true;
                }
                if state.declared && !state.groups.contains_key(&tag.group_idx) {
                    return true;
                }
                if let Some(count) = tag.expected_group_count
                    && (count == 0
                        || state
                            .expected_group_count
                            .is_some_and(|expected| expected != count as usize))
                {
                    return true;
                }
                if let Some(alignment) = tag.alignment_blocks {
                    let alignment = alignment.max(1) as usize;
                    if state.alignment_blocks != 0 && state.alignment_blocks != alignment {
                        return true;
                    }
                }
                let layout_changed = !state.groups.contains_key(&tag.group_idx)
                    || tag
                        .expected_group_count
                        .is_some_and(|count| state.expected_group_count != Some(count as usize))
                    || tag.alignment_blocks.is_some_and(|alignment| {
                        state.alignment_blocks.max(1) != alignment.max(1) as usize
                    });
                let previous =
                    layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
                if let Some(count) = tag.expected_group_count {
                    state.expected_group_count = Some(count as usize);
                }
                if let Some(alignment) = tag.alignment_blocks {
                    state.alignment_blocks = alignment.max(1) as usize;
                }
                state.groups.entry(tag.group_idx).or_default();
                if layout_changed {
                    state.rebuild_complete_checkpoints();
                }
                let mut updates = Vec::new();
                for block in &data.blocks {
                    let hash = block.block_hash.0;
                    let old = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    state.store(tag.group_idx, block.block_hash.0, event.storage_tier);
                    let new = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    if !layout_changed && old != new {
                        updates.push((hash, new));
                    }
                }
                if let Some(previous) = previous {
                    state.refresh_general_requirement(&self.general_device_eval_layouts);
                    self.publish_reset(owner, previous, &state);
                } else {
                    self.checkpoints
                        .publish(owner, updates, state.alignment_blocks.max(1));
                }
            }
            KvCacheEventData::Removed(data) => {
                if let Some(state) = self.owner(owner) {
                    let mut state = state.write();
                    if state.retired {
                        return true;
                    }
                    let mut updates = Vec::new();
                    for hash in &data.block_hashes {
                        let old = state
                            .complete_checkpoints
                            .get(&hash.0)
                            .copied()
                            .unwrap_or(0);
                        state.remove(tag.group_idx, hash.0, event.storage_tier);
                        let new = state
                            .complete_checkpoints
                            .get(&hash.0)
                            .copied()
                            .unwrap_or(0);
                        if old != new {
                            updates.push((hash.0, new));
                        }
                    }
                    self.checkpoints
                        .publish(owner, updates, state.alignment_blocks.max(1));
                }
            }
            KvCacheEventData::Cleared => self.clear_event(event),
        }
        true
    }

    pub fn clear_event(&self, event: &RouterEvent) {
        let Ok(Some(scope)) = event.reset_scope() else {
            return;
        };
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let registry = self.registry.load();
        for (owner, state) in &registry.owners {
            let matches = match (scope, *owner) {
                (
                    ResetScope::All | ResetScope::Domain(ResidencyDomain::Worker),
                    ResidencyOwner::Worker(w),
                ) => w == worker,
                (
                    ResetScope::All | ResetScope::Domain(ResidencyDomain::CacheOwner),
                    ResidencyOwner::CacheOwner(source),
                ) => event.state_source == Some(source),
                _ => false,
            };
            if matches {
                // A cache clear preserves the declared layout; it only invalidates checkpoints.
                self.clear_owner(*owner, &mut state.write());
            }
        }
    }

    pub fn clear_worker(&self, worker: WorkerWithDpRank) {
        if let Some(state) = self.owner(ResidencyOwner::worker(worker)) {
            self.clear_owner(ResidencyOwner::worker(worker), &mut state.write());
        }
    }

    pub fn remove_worker_rank(&self, worker: WorkerWithDpRank) {
        self.retire_workers(|known| known == worker);
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.retire_workers(|worker| worker.worker_id == worker_id);
    }

    fn retire_workers(&self, matches: impl Fn(WorkerWithDpRank) -> bool) {
        let _writer = self.registry_writer.lock();
        let current = self.registry.load_full();
        let mut next = (*current).clone();
        next.owners.retain(|owner, state| {
            if matches!(owner, ResidencyOwner::Worker(worker) if matches(*worker)) {
                let mut state = state.write();
                state.retired = true;
                self.clear_owner(*owner, &mut state);
                false
            } else {
                true
            }
        });
        self.active
            .store(!next.owners.is_empty(), Ordering::Relaxed);
        self.registry.store(Arc::new(next));
    }

    /// Dump empty groups too: absence of a checkpoint must survive recovery.
    pub fn dump_events(&self) -> Vec<RouterEvent> {
        let registry = self.registry.load();
        let mut out = Vec::new();
        for (&owner, state) in &registry.owners {
            let state = state.read();
            for (&group_idx, group) in &state.groups {
                for tier in [
                    StorageTier::Device,
                    StorageTier::HostPinned,
                    StorageTier::Disk,
                    StorageTier::External,
                ] {
                    if matches!(owner, ResidencyOwner::CacheOwner(_)) && tier == StorageTier::Device
                    {
                        continue;
                    }
                    let blocks: Vec<_> = group
                        .iter()
                        .filter(|(_, mask)| **mask & tier_bit(tier) != 0)
                        .map(|(&hash, _)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(0),
                            mm_extra_info: None,
                        })
                        .collect();
                    if blocks.is_empty()
                        && tier
                            != match owner {
                                ResidencyOwner::Worker(_) => StorageTier::Device,
                                _ => StorageTier::HostPinned,
                            }
                    {
                        continue;
                    }
                    let event = KvCacheEvent {
                        event_id: 0,
                        dp_rank: match owner {
                            ResidencyOwner::Worker(w) => w.dp_rank,
                            _ => 0,
                        },
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: None,
                            blocks,
                        }),
                    };
                    let event = match owner {
                        ResidencyOwner::Worker(w) => {
                            RouterEvent::with_storage_tier(w.worker_id, event, tier)
                        }
                        ResidencyOwner::CacheOwner(source) => {
                            RouterEvent::with_cache_owner(0, event, tier, source)
                        }
                    };
                    out.push(event.with_kv_cache_group(KvCacheGroupTag {
                        group_idx,
                        kind: KvCacheGroupKind::Mamba,
                        expected_group_count: state.expected_group_count.map(|count| count as u32),
                        alignment_blocks: Some(state.alignment_blocks.max(1) as u32),
                    }));
                }
            }
        }
        out
    }

    pub fn is_hybrid(&self, worker: WorkerWithDpRank) -> bool {
        self.owner(ResidencyOwner::worker(worker)).is_some()
    }

    #[cfg(test)]
    pub(crate) fn groups_for(&self, worker: WorkerWithDpRank) -> rustc_hash::FxHashSet<u32> {
        self.owner(ResidencyOwner::worker(worker))
            .map(|state| state.read().groups.keys().copied().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::KvCacheRemoveData;

    fn checkpoint(
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group: u32,
        hash: u64,
        count: Option<u32>,
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                dp_rank: worker.dp_rank,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(0),
                        mm_extra_info: None,
                    }],
                }),
            },
            tier,
        )
        .with_kv_cache_group(KvCacheGroupTag {
            group_idx: group,
            kind: KvCacheGroupKind::Mamba,
            expected_group_count: count,
            alignment_blocks: Some(1),
        })
    }

    fn has_checkpoint(
        table: &KvCacheGroupResidency,
        owner: ResidencyOwner,
        hash: u64,
        tier: StorageTier,
    ) -> bool {
        let mut view = table.read_view();
        view.track_owner(owner);
        view.owners_at(ExternalSequenceBlockHash(hash), tier)
            .any(|candidate| candidate == owner)
    }

    #[test]
    fn malformed_zero_group_count_keeps_an_incomplete_layout() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(0)));
        assert!(
            !table
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn incomplete_membership_and_empty_groups_survive_dump() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_groups(owner, &[0, 1, 2]);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(3)));
        }
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        assert_eq!(restored.groups_for(worker).len(), 3);
        assert!(!has_checkpoint(&restored, owner, 11, StorageTier::Device));
        restored.apply(&checkpoint(worker, StorageTier::Device, 2, 11, Some(3)));
        assert!(has_checkpoint(&restored, owner, 11, StorageTier::Device));
    }

    #[test]
    fn tier_membership_and_removal_are_independent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let chain = [ExternalSequenceBlockHash(11)];
        let mut event = checkpoint(worker, StorageTier::HostPinned, 0, 11, Some(1));
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
        event.storage_tier = StorageTier::Device;
        table.apply(&event);
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: chain.to_vec(),
        });
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
    }

    #[test]
    fn missing_completeness_metadata_fails_closed() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, None));
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn cache_clear_preserves_layout_but_worker_retirement_removes_it() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.declare_groups(ResidencyOwner::worker(worker), &[0, 1]);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(2)));
        let clear = RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                dp_rank: 0,
                data: KvCacheEventData::Cleared,
            },
        );
        table.clear_event(&clear);
        assert_eq!(table.groups_for(worker).len(), 2);
        assert!(!has_checkpoint(
            &table,
            ResidencyOwner::worker(worker),
            11,
            StorageTier::Device
        ));
        table.remove_worker(7);
        assert!(!table.is_active());
    }

    #[test]
    fn unknown_group_kind_is_consumed_without_mutation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 11, Some(1));
        event.kv_cache_group.as_mut().unwrap().kind = KvCacheGroupKind::Unknown;
        assert!(table.apply(&event));
        assert!(!table.is_active());
    }

    #[test]
    fn read_view_only_exposes_complete_tiers_and_duplicate_events_are_idempotent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0, 1], 1);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 11, Some(2));
        table.apply(&event);
        table.apply(&event);
        let mut incomplete = table.read_view();
        assert!(incomplete.track_owner(owner).unwrap().complete_layout);
        assert!(
            incomplete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        table.apply(&checkpoint(worker, StorageTier::Device, 1, 11, Some(2)));
        for group in [0, 1] {
            table.apply(&checkpoint(
                worker,
                StorageTier::HostPinned,
                group,
                11,
                Some(2),
            ));
        }
        let mut complete = table.read_view();
        complete.track_owner(owner);
        complete.track_owner(owner);
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash, hash],
        });
        table.apply(&event);
        table.apply(&event);
        let mut removed = table.read_view();
        removed.track_owner(owner);
        assert!(
            removed
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        assert_eq!(
            removed
                .owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert!(
            complete
                .owners_at(hash, StorageTier::Device)
                .next()
                .is_none()
        );
    }

    #[test]
    fn layout_declaration_and_changes_rebuild_complete_membership() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, None));
        }
        let mut incomplete = table.read_view();
        assert!(!incomplete.track_owner(owner).unwrap().complete_layout);
        assert!(
            incomplete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        table.declare_layout(owner, &[0, 1], 1);
        table.declare_layout(owner, &[0, 1], 1);
        let mut declared = table.read_view();
        declared.track_owner(owner);
        assert_eq!(
            declared
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        table.declare_layout(owner, &[0, 1, 2], 1);
        let mut changed = table.read_view();
        assert!(changed.track_owner(owner).unwrap().complete_layout);
        assert!(
            changed
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        for group in [0, 1, 2] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(3)));
        }
        table.apply(&checkpoint(worker, StorageTier::Device, 9, 11, Some(3)));
        let mut repopulated = table.read_view();
        repopulated.track_owner(owner);
        assert_eq!(
            repopulated
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        table.declare_layout(owner, &[0, 1, 2], 2);
        let mut realigned = table.read_view();
        assert_eq!(realigned.track_owner(owner).unwrap().alignment_blocks, 2);
        assert!(
            realigned
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
    }

    #[test]
    fn read_view_tracks_only_requested_owners_from_its_registry_snapshot() {
        let table = KvCacheGroupResidency::new();
        let a = WorkerWithDpRank::new(7, 0);
        let b = WorkerWithDpRank::new(8, 0);
        let later = WorkerWithDpRank::new(9, 0);
        for worker in [a, b] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        }
        let mut view = table.read_view();
        assert!(view.layout(ResidencyOwner::worker(a)).is_none());
        view.track_owner(ResidencyOwner::worker(a));
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[ResidencyOwner::worker(a)]
        );
        assert!(view.layout(ResidencyOwner::worker(b)).is_none());
        view.track_owner(ResidencyOwner::worker(b));
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[ResidencyOwner::worker(a), ResidencyOwner::worker(b)]
        );
        table.apply(&checkpoint(later, StorageTier::Device, 0, 11, Some(1)));
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
    }

    #[test]
    fn complete_membership_survives_dump_and_clear_preserves_only_layout() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, &[0, 1], 1);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(2)));
            table.apply(&checkpoint(
                worker,
                StorageTier::HostPinned,
                group,
                22,
                Some(2),
            ));
        }
        table.apply(&checkpoint(worker, StorageTier::Disk, 0, 33, Some(2)));
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        let mut view = restored.read_view();
        view.track_owner(owner);
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(22), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert!(
            view.owners_at(ExternalSequenceBlockHash(33), StorageTier::Disk)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        restored.clear_worker(worker);
        let mut cleared = restored.read_view();
        assert!(cleared.track_owner(owner).unwrap().complete_layout);
        assert!(
            cleared
                .owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        assert!(
            cleared
                .owners_at(ExternalSequenceBlockHash(22), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        restored.remove_worker_rank(worker);
        assert!(restored.read_view().track_owner(owner).is_none());
        assert!(!restored.is_active());
    }

    #[test]
    fn compact_cache_owner_lookup_and_clear_do_not_change_worker_membership() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };

        let source = CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        );
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let worker_owner = ResidencyOwner::worker(worker);
        let cache_owner = ResidencyOwner::cache_owner(source);
        for group in [0, 1] {
            let event = checkpoint(worker, StorageTier::HostPinned, group, 11, Some(2));
            table.apply(&event);
            let mut event =
                RouterEvent::with_cache_owner(7, event.event, StorageTier::HostPinned, source);
            event.kv_cache_group = Some(KvCacheGroupTag {
                group_idx: group,
                kind: KvCacheGroupKind::Mamba,
                expected_group_count: Some(2),
                alignment_blocks: Some(1),
            });
            table.apply(&event);
        }
        let mut view = table.read_view();
        assert_eq!(
            view.owner_for_key(cache_owner.compact_key()),
            Some(cache_owner)
        );
        assert_eq!(
            view.track_cache_owner(cache_owner.compact_key()),
            Some(cache_owner)
        );
        assert_eq!(
            view.track_cache_owner(cache_owner.compact_key()),
            Some(cache_owner)
        );
        view.track_owner(worker_owner);
        let owners = view
            .owners_at(ExternalSequenceBlockHash(11), StorageTier::HostPinned)
            .collect::<Vec<_>>();
        assert_eq!(owners.len(), 2);
        assert!(owners.contains(&cache_owner));
        assert!(owners.contains(&worker_owner));

        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        restored.clear_event(&RouterEvent::with_cache_owner(
            0,
            KvCacheEvent {
                event_id: 2,
                dp_rank: 0,
                data: KvCacheEventData::Cleared,
            },
            StorageTier::HostPinned,
            source,
        ));
        let mut cleared = restored.read_view();
        cleared.track_cache_owner(cache_owner.compact_key());
        cleared.track_owner(worker_owner);
        assert!(cleared.layout(cache_owner).unwrap().complete_layout);
        assert_eq!(
            cleared
                .owners_at(ExternalSequenceBlockHash(11), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[worker_owner]
        );
    }

    #[test]
    fn direct_workers_require_complete_groups_and_alignment_one() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(ResidencyOwner::worker(first), &[0, 1], 1);
        table.declare_layout(ResidencyOwner::worker(second), &[0], 2);
        table.apply(&checkpoint(first, StorageTier::Device, 0, 11, Some(2)));
        assert!(!table.with_checkpoint_workers(hash, |_| panic!("incomplete state")));
        table.apply(&checkpoint(first, StorageTier::Device, 1, 11, Some(2)));
        let mut aligned = checkpoint(second, StorageTier::Device, 0, 11, Some(1));
        aligned.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(2);
        table.apply(&aligned);
        assert!(table.requires_general_device_eval());
        assert!(table.with_checkpoint_workers(hash, |workers| {
            assert_eq!(workers.len(), 1);
            assert!(workers.contains(&first));
            assert!(!workers.contains(&second));
        }));
        table.remove_worker_rank(second);
        assert!(!table.requires_general_device_eval());
    }

    #[test]
    fn current_records_observe_remove_readd_and_reclaim_empty_entries() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let mut view = table.read_view();
        view.track_owner(owner);
        let mut retained_workers = None;
        table.with_checkpoint_workers(hash, |workers| retained_workers = Some(workers.clone()));
        table.clear_worker(worker);
        assert!(view.owners_at(hash, StorageTier::Device).next().is_none());
        assert!(table.checkpoints.entries.is_empty());
        assert!(retained_workers.unwrap().contains(&worker));
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            vec![owner]
        );
    }

    #[test]
    fn known_worker_classification_does_not_learn_later_registration() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let known = table.known_worker_read();
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        assert!(!known.is_known_worker(worker));
        assert!(table.known_worker_read().is_known_worker(worker));
    }

    #[test]
    fn retirement_after_view_capture_keeps_owner_known() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let known = table.known_worker_read();
        let mut view = table.read_view();
        table.remove_worker_rank(worker);
        assert!(known.is_known_worker(worker));
        assert!(!view.track_owner(owner).unwrap().complete_layout);
        assert!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .next()
                .is_none()
        );
        assert!(table.read_view().capture_layout(owner).is_none());
    }

    #[test]
    fn retired_owner_handle_cannot_publish_after_readd() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let old = table.owner(owner).unwrap();
        table.remove_worker_rank(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 22, Some(1)));
        assert!(old.read().retired);
        assert!(!Arc::ptr_eq(&old, &table.owner(owner).unwrap()));
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 22, StorageTier::Device));
        table.clear_owner(owner, &mut old.write());
        assert!(has_checkpoint(&table, owner, 22, StorageTier::Device));
    }

    #[test]
    fn concurrent_owner_updates_and_retirement_leave_no_detached_writes() {
        use std::sync::Barrier;
        use std::thread;

        let table = Arc::new(KvCacheGroupResidency::new());
        let barrier = Arc::new(Barrier::new(3));
        let mut writers = Vec::new();
        for id in [7, 8] {
            let table = table.clone();
            let barrier = barrier.clone();
            writers.push(thread::spawn(move || {
                let worker = WorkerWithDpRank::new(id, 0);
                barrier.wait();
                for hash in 0..128 {
                    table.apply(&checkpoint(worker, StorageTier::Device, 0, hash, Some(1)));
                    if hash % 7 == 0 {
                        table.remove_worker_rank(worker);
                    }
                }
                table.remove_worker_rank(worker);
            }));
        }
        barrier.wait();
        for _ in 0..256 {
            table.with_checkpoint_workers(ExternalSequenceBlockHash(11), |_| {});
        }
        for writer in writers {
            writer.join().unwrap();
        }
        assert!(!table.is_active());
        assert!(table.checkpoints.entries.is_empty());
    }

    #[test]
    fn current_membership_keeps_dp_ranks_and_tiers_separate() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        let hash = ExternalSequenceBlockHash(11);
        for worker in [first, second] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        }
        table.apply(&checkpoint(first, StorageTier::HostPinned, 0, 11, Some(1)));
        table.with_checkpoint_workers(hash, |workers| {
            assert_eq!(workers.len(), 2);
            assert!(workers.contains(&first));
            assert!(workers.contains(&second));
        });
        let view = table.read_view();
        assert_eq!(
            view.owner_layouts_at(hash, StorageTier::HostPinned)
                .map(|(owner, _)| owner)
                .collect::<Vec<_>>(),
            vec![ResidencyOwner::worker(first)]
        );
        table.remove_worker_rank(first);
        assert!(table.with_checkpoint_workers(hash, |workers| {
            assert_eq!(workers.len(), 1);
            assert!(workers.contains(&second));
        }));
        assert!(
            view.owner_layouts_at(hash, StorageTier::HostPinned)
                .next()
                .is_none()
        );
        table.remove_worker(7);
        assert!(table.checkpoints.entries.is_empty());
    }

    #[test]
    fn general_gate_tracks_alignment_through_clear_remap_and_readd() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, &[0], 2);
        assert!(table.requires_general_device_eval());
        table.clear_worker(worker);
        assert!(table.requires_general_device_eval());
        table.declare_layout(owner, &[0], 1);
        assert!(!table.requires_general_device_eval());
        table.declare_layout(owner, &[0], 2);
        table.remove_worker_rank(worker);
        assert!(!table.requires_general_device_eval());
        table.declare_layout(owner, &[0], 1);
        assert!(!table.requires_general_device_eval());
    }

    #[test]
    fn concurrent_same_owner_store_and_retirement_leave_no_checkpoints() {
        use std::sync::Barrier;
        use std::thread;

        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let barrier = Barrier::new(3);
        thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                for hash in 0..256 {
                    table.apply(&checkpoint(worker, StorageTier::Device, 0, hash, Some(1)));
                }
            });
            scope.spawn(|| {
                barrier.wait();
                for _ in 0..128 {
                    table.remove_worker_rank(worker);
                }
            });
            barrier.wait();
            for hash in 0..256 {
                table.with_checkpoint_workers(ExternalSequenceBlockHash(hash), |_| {});
            }
        });
        table.remove_worker_rank(worker);
        assert!(!table.is_active());
        assert!(table.checkpoints.entries.is_empty());
    }
}
