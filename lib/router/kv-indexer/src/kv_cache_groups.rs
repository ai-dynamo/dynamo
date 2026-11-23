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
    atomic::{AtomicBool, Ordering},
};

use arc_swap::ArcSwap;
use dashmap::{DashMap, mapref::entry::Entry};
use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::prelude::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, ResidencyOwnerKeyExt, RouterEvent, StorageTier, WorkerId,
    WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;
type TierOwners = FxHashSet<ResidencyOwner>;

/// One current checkpoint record. Group completeness is resolved on the write
/// side; readers need neither owner-layout joins nor publication certificates.
#[derive(Debug, Default)]
struct CheckpointPresence {
    owners: [TierOwners; 4],
}

impl CheckpointPresence {
    fn update(&mut self, owner: ResidencyOwner, mask: u8) {
        for (index, owners) in self.owners.iter_mut().enumerate() {
            if mask & (1 << index) != 0 {
                owners.insert(owner);
            } else {
                owners.remove(&owner);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.owners.iter().all(FxHashSet::is_empty)
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

    fn publish(&self, owner: ResidencyOwner, updates: Vec<(u64, u8)>) {
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
                        record.update(owner, mask);
                        record.is_empty()
                    };
                    if empty {
                        entry.remove();
                    }
                }
                Entry::Vacant(entry) if mask != 0 => {
                    let mut record = CheckpointPresence::default();
                    record.update(owner, mask);
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
        visit: impl FnOnce(&TierOwners),
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
    retired: bool,
    /// Mirror of `layout().complete_layout` that queries load without this lock.
    published_layout: Arc<AtomicBool>,
}

impl OwnerPresence {
    /// Call after any change to the group set, the expected count or retirement.
    fn publish_layout(&self) {
        self.published_layout
            .store(self.layout().complete_layout, Ordering::Relaxed);
    }

    fn layout(&self) -> CheckpointLayout {
        CheckpointLayout {
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
    pub complete_layout: bool,
}

/// Only registration topology is shared as a snapshot. Checkpoint data lives in
/// independent mutable records and has the same best-effort freshness as KV facts.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, OwnerEntry>,
    cache_owners: FxHashMap<ResidencyOwnerKey, ResidencyOwner>,
}

#[derive(Debug, Clone)]
struct OwnerEntry {
    state: Arc<RwLock<OwnerPresence>>,
    complete_layout: Arc<AtomicBool>,
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
        let entry = self.registry.owners.get(&owner)?;
        // A registered owner remains known to this query even if retirement
        // happens before its first layout read. Its retired layout is incomplete,
        // so absent checkpoints receive zero credit rather than dense KV credit.
        // The owner lock is not taken: writers hold it across republication.
        Some(CheckpointLayout {
            complete_layout: entry.complete_layout.load(Ordering::Relaxed),
        })
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

    /// The callback receives the current complete owners from one record read.
    /// It must not acquire owner locks or recursively enter the directory while
    /// holding this record read.
    pub fn with_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&FxHashSet<ResidencyOwner>),
    ) -> bool {
        self.checkpoints.with_owners(hash, tier, visit)
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
                    .iter()
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
}

impl KvCacheGroupResidency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub fn read_view(&self) -> CheckpointReadView {
        CheckpointReadView {
            registry: self.registry.load_full(),
            layouts: FxHashMap::default(),
            checkpoints: self.checkpoints.clone(),
        }
    }

    fn owner(&self, owner: ResidencyOwner) -> Option<Arc<RwLock<OwnerPresence>>> {
        self.registry
            .load()
            .owners
            .get(&owner)
            .map(|entry| entry.state.clone())
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
        self.checkpoints.publish(owner, updates);
    }

    fn clear_owner(&self, owner: ResidencyOwner, state: &mut OwnerPresence) {
        let previous = std::mem::take(&mut state.complete_checkpoints);
        state.clear_checkpoints();
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
        if let Some(entry) = current.owners.get(&owner) {
            return entry.state.clone();
        }
        let mut next = (*current).clone();
        let presence = OwnerPresence::default();
        let complete_layout = presence.published_layout.clone();
        let state = Arc::new(RwLock::new(presence));
        next.owners.insert(
            owner,
            OwnerEntry {
                state: state.clone(),
                complete_layout,
            },
        );
        if matches!(owner, ResidencyOwner::CacheOwner(_)) {
            next.cache_owners.insert(owner.compact_key(), owner);
        }
        self.registry.store(Arc::new(next));
        self.active.store(true, Ordering::Relaxed);
        state
    }

    /// Only an alignment of one block is supported. Any other value keeps the
    /// owner known as hybrid with no usable checkpoint.
    pub fn declare_layout(&self, owner: ResidencyOwner, groups: &[u32], alignment_blocks: u32) {
        if alignment_blocks > 1 && !groups.is_empty() {
            self.close_layout(owner, alignment_blocks);
            return;
        }
        self.install_layout(owner, groups);
    }

    fn close_layout(&self, owner: ResidencyOwner, alignment_blocks: u32) {
        let state = self.get_or_create(owner);
        let mut state = state.write();
        if state.retired || (state.declared && state.groups.is_empty()) {
            return;
        }
        tracing::warn!(
            ?owner,
            alignment_blocks,
            "unsupported recurrent checkpoint alignment; the owner gets no checkpoint credit"
        );
        let previous = std::mem::take(&mut state.complete_checkpoints);
        state.groups.clear();
        state.group_counts.clear();
        state.declared = true;
        state.expected_group_count = None;
        state.publish_layout();
        self.publish_reset(owner, previous, &state);
    }

    fn install_layout(&self, owner: ResidencyOwner, groups: &[u32]) {
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
            || previous_expected_count != Some(groups.len());
        let previous = layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
        let changed = state.declared
            && (state.groups.len() != groups.len()
                || !groups.iter().all(|group| state.groups.contains_key(group)));
        if changed {
            state.groups.clear();
        }
        state.groups.retain(|group, _| groups.contains(group));
        for &group in groups {
            state.groups.entry(group).or_default();
        }
        state.declared = true;
        state.expected_group_count = Some(groups.len());
        state.publish_layout();
        if changed || removed_groups {
            state.rebuild_group_counts();
        } else if layout_changed {
            state.rebuild_complete_checkpoints();
        }
        if let Some(previous) = previous {
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
                if tag.alignment_blocks.is_some_and(|alignment| alignment > 1) {
                    return true;
                }
                let layout_changed = !state.groups.contains_key(&tag.group_idx)
                    || tag
                        .expected_group_count
                        .is_some_and(|count| state.expected_group_count != Some(count as usize));
                let previous =
                    layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
                if let Some(count) = tag.expected_group_count {
                    state.expected_group_count = Some(count as usize);
                }
                state.groups.entry(tag.group_idx).or_default();
                state.publish_layout();
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
                    self.publish_reset(owner, previous, &state);
                } else {
                    self.checkpoints.publish(owner, updates);
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
                    self.checkpoints.publish(owner, updates);
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
        for (owner, entry) in &registry.owners {
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
                self.clear_owner(*owner, &mut entry.state.write());
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
        next.owners.retain(|owner, entry| {
            if matches!(owner, ResidencyOwner::Worker(worker) if matches(*worker)) {
                let mut state = entry.state.write();
                state.retired = true;
                state.publish_layout();
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
        for (&owner, entry) in &registry.owners {
            let state = entry.state.read();
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
                        alignment_blocks: Some(1),
                    }));
                }
            }
        }
        out
    }

    pub fn is_hybrid(&self, worker: WorkerWithDpRank) -> bool {
        self.registry
            .load()
            .owners
            .contains_key(&ResidencyOwner::worker(worker))
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
    use crate::prelude::KvCacheRemoveData;

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
        table.declare_layout(owner, &[0, 1, 2], 1);
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
        table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn cache_clear_preserves_layout_but_worker_retirement_removes_it() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.declare_layout(ResidencyOwner::worker(worker), &[0, 1], 1);
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
        let mut unsupported = table.read_view();
        assert!(!unsupported.track_owner(owner).unwrap().complete_layout);
        assert!(
            unsupported
                .owners_at(hash, StorageTier::Device)
                .next()
                .is_none()
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
        use dynamo_router_protocol::identity::{
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
    fn unsupported_alignment_keeps_the_owner_hybrid_without_checkpoints() {
        let table = KvCacheGroupResidency::new();
        let declared = WorkerWithDpRank::new(7, 0);
        let tagged = WorkerWithDpRank::new(7, 1);
        let owner = ResidencyOwner::worker(declared);
        table.declare_layout(owner, &[0], 2);
        table.apply(&checkpoint(declared, StorageTier::Device, 0, 11, Some(1)));
        let mut event = checkpoint(tagged, StorageTier::Device, 0, 11, Some(1));
        event.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(2);
        assert!(table.apply(&event));
        for worker in [declared, tagged] {
            assert!(table.is_hybrid(worker));
            assert!(!has_checkpoint(
                &table,
                ResidencyOwner::worker(worker),
                11,
                StorageTier::Device
            ));
        }
        assert!(table.checkpoints.entries.is_empty());

        table.declare_layout(owner, &[0], 1);
        table.apply(&checkpoint(declared, StorageTier::Device, 0, 11, Some(1)));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
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
        table.clear_worker(worker);
        assert!(view.owners_at(hash, StorageTier::Device).next().is_none());
        assert!(table.checkpoints.entries.is_empty());
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            vec![owner]
        );
    }

    #[test]
    fn retirement_after_view_capture_keeps_owner_known() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let mut view = table.read_view();
        table.remove_worker_rank(worker);
        assert!(!view.track_owner(owner).unwrap().complete_layout);
        assert!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .next()
                .is_none()
        );
        assert!(table.read_view().capture_layout(owner).is_none());
    }

    #[test]
    fn layout_capture_does_not_wait_for_the_owner_lock() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let state = table.owner(owner).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let _held = state.write();
            scope.spawn(|| sender.send(table.read_view().track_owner(owner)).unwrap());
            let layout = receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("layout capture waited for the owner lock");
            assert!(layout.unwrap().complete_layout);
        });
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
            table.read_view().with_owners(
                ExternalSequenceBlockHash(11),
                StorageTier::Device,
                |_| {},
            );
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
        let mut view = table.read_view();
        for worker in [first, second] {
            view.track_owner(ResidencyOwner::worker(worker));
        }
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            vec![
                ResidencyOwner::worker(first),
                ResidencyOwner::worker(second)
            ]
        );
        assert_eq!(
            view.owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>(),
            vec![ResidencyOwner::worker(first)]
        );
        table.remove_worker_rank(first);
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            vec![ResidencyOwner::worker(second)]
        );
        assert!(
            view.owners_at(hash, StorageTier::HostPinned)
                .next()
                .is_none()
        );
        table.remove_worker(7);
        assert!(table.checkpoints.entries.is_empty());
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
                table.read_view().with_owners(
                    ExternalSequenceBlockHash(hash),
                    StorageTier::Device,
                    |_| {},
                );
            }
        });
        table.remove_worker_rank(worker);
        assert!(!table.is_active());
        assert!(table.checkpoints.entries.is_empty());
    }

    #[test]
    fn vllm_checkpoint_only_payload_reaches_presence_and_removes_without_tokens() {
        use dynamo_kv_events::zmq_wire::{RawKvEvent, ZmqEventNormalizer, decode_event_batch};
        use rmp_serde::from_slice;

        let payload = serde_json::json!([0.0, [{
            "type": "BlockStored", "block_hashes": [333], "parent_block_hash": 222,
            "token_ids": [], "block_size": 576, "medium": "CPU", "group_idx": 0,
            "kv_cache_spec_kind": "mamba"
        }], 0]);
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        let mut decoded = decode_event_batch(&wire).unwrap();
        let worker = WorkerWithDpRank::new(7, 0);
        let raw = decoded.events.remove(0);
        assert!(
            ZmqEventNormalizer::new(576)
                .preprocess(raw.clone(), worker)
                .is_none(),
            "default publisher must not emit tagged events"
        );
        let mut normalizer = ZmqEventNormalizer::new(576).with_recurrent_groups(Some(vec![0]));
        let store = normalizer
            .normalize(raw, 1, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        let KvCacheEventData::Stored(data) = &store.event.data else {
            panic!("store");
        };
        assert_eq!(data.blocks.len(), 1);
        assert_eq!(data.blocks[0].block_hash, ExternalSequenceBlockHash(333));
        assert!(!store.targets_primary().unwrap());
        let presence = KvCacheGroupResidency::new();
        presence.apply(&store);
        let has_checkpoint = || {
            let mut view = presence.read_view();
            let owner = store.residency_owner().unwrap();
            view.track_owner(owner);
            view.owners_at(ExternalSequenceBlockHash(333), StorageTier::HostPinned)
                .any(|known| known == owner)
        };
        assert!(has_checkpoint());
        let remove: RawKvEvent = from_slice(&rmp_serde::to_vec_named(&serde_json::json!({"type":"BlockRemoved", "block_hashes":[333], "medium":"CPU", "group_idx":0})).unwrap()).unwrap();
        let remove = normalizer
            .normalize(remove, 2, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        presence.apply(&remove);
        assert!(!has_checkpoint());
    }

    /// SGLang publishes `format=dynamo` map events: untagged attention pages with
    /// token ids, and (sgl-project/sglang#51647) one tagged Mamba store per cached
    /// state at the node's last page hash, plus a tagged removal on tombstone.
    fn sglang_stream() -> Vec<dynamo_kv_events::zmq_wire::RawKvEvent> {
        use dynamo_kv_events::zmq_wire::decode_event_batch;

        let payload = serde_json::json!([0.0, [
            {
                "type": "BlockStored", "block_hashes": [11, 12, 13, 14],
                "parent_block_hash": null, "token_ids": (0..64).collect::<Vec<u32>>(),
                "block_size": 16, "lora_id": null, "medium": "GPU",
                "cache_salt": null, "session_id": null, "lora_name": null
            },
            {
                "type": "BlockStored", "block_hashes": [14], "parent_block_hash": 13,
                "token_ids": [], "block_size": 16, "lora_id": null, "medium": "GPU",
                "group_idx": 1, "kv_cache_spec_kind": "mamba"
            },
            {
                "type": "BlockRemoved", "block_hashes": [14], "medium": "GPU",
                "group_idx": 1, "kv_cache_spec_kind": "mamba"
            }
        ], 0]);
        let wire = rmp_serde::to_vec_named(&payload).unwrap();
        decode_event_batch(&wire).unwrap().events
    }

    #[test]
    fn sglang_last_page_checkpoint_tracks_store_and_tombstone() {
        use dynamo_kv_events::zmq_wire::ZmqEventNormalizer;

        let worker = WorkerWithDpRank::new(9, 0);
        let owner = ResidencyOwner::worker(worker);
        let mut events = sglang_stream().into_iter();
        let (attention, store, tombstone) = (
            events.next().unwrap(),
            events.next().unwrap(),
            events.next().unwrap(),
        );

        // Without the opt-in the tagged events are dropped and the attention
        // pages are the only thing the router sees, as before.
        let mut dense = ZmqEventNormalizer::new(16);
        assert!(dense.normalize(attention.clone(), 1, worker).is_some());
        assert!(dense.preprocess(store.clone(), worker).is_none());

        let mut normalizer = ZmqEventNormalizer::new(16)
            .with_recurrent_groups(Some(vec![1]))
            .with_recurrent_alignment_blocks(1);
        let attention = normalizer
            .normalize(attention, 1, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        assert!(attention.targets_primary().unwrap());
        let KvCacheEventData::Stored(pages) = &attention.event.data else {
            panic!("attention store");
        };
        assert_eq!(pages.blocks.len(), 4);
        assert_eq!(pages.blocks[3].block_hash, ExternalSequenceBlockHash(14));

        let presence = KvCacheGroupResidency::new();
        presence.declare_layout(owner, &[1], 1);
        let store = normalizer
            .normalize(store, 2, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        assert!(!store.targets_primary().unwrap());
        assert_eq!(store.residency_owner().unwrap(), owner);
        presence.apply(&store);
        let checkpoint_at = |hash: u64| {
            let mut view = presence.read_view();
            view.track_owner(owner);
            view.owners_at(ExternalSequenceBlockHash(hash), StorageTier::Device)
                .any(|known| known == owner)
        };
        assert!(presence.is_hybrid(worker));
        assert!(checkpoint_at(14), "the state sits at the node's last page");
        assert!(
            !checkpoint_at(13),
            "earlier pages of the node hold no state"
        );

        let tombstone = normalizer
            .normalize(tombstone, 3, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        presence.apply(&tombstone);
        assert!(presence.is_hybrid(worker));
        assert!(!checkpoint_at(14), "a tombstone withdraws the checkpoint");
    }

    #[test]
    fn sglang_page_smaller_than_the_mamba_chunk_gets_no_credit() {
        use dynamo_kv_events::zmq_wire::ZmqEventNormalizer;

        // page_size 16 under a 64-token Mamba chunk would need alignment 4. The
        // table supports alignment 1 only, so the worker stays hybrid with no
        // usable checkpoint; the publisher refuses this geometry at startup.
        let worker = WorkerWithDpRank::new(9, 0);
        let owner = ResidencyOwner::worker(worker);
        let mut events = sglang_stream().into_iter();
        let _attention = events.next().unwrap();
        let store = events.next().unwrap();
        let mut normalizer = ZmqEventNormalizer::new(16)
            .with_recurrent_groups(Some(vec![1]))
            .with_recurrent_alignment_blocks(4);
        let presence = KvCacheGroupResidency::new();
        presence.declare_layout(owner, &[1], 4);
        let store = normalizer
            .normalize(store, 1, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        presence.apply(&store);
        let mut view = presence.read_view();
        view.track_owner(owner);
        assert!(presence.is_hybrid(worker));
        assert!(
            !view
                .owners_at(ExternalSequenceBlockHash(14), StorageTier::Device)
                .any(|known| known == owner),
            "unsupported alignment keeps the owner known but uncredited"
        );
    }
}
