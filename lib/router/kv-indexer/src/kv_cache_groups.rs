// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checkpoint residency for hybrid (attention + recurrent state) models.
//!
//! For the main-attention group a positional block hash names a reusable
//! block. For the recurrent (Mamba / GDN / linear-attention) groups it names
//! the state checkpoint at that boundary. A hybrid engine can resume a prefix
//! only at a boundary where every recurrent group holds its state, so an
//! attention-only match overestimates the reusable prefix and over-sizes KV
//! transfer hints (ai-dynamo/dynamo#16794).
//!
//! The worker joins the engine's per-group events and publishes one tagged
//! event per complete checkpoint (`ZmqEventNormalizer` in `dynamo-kv-events`).
//! This table records which owner holds which checkpoint in which tier.
//! Tagged events update that membership, never attention edges. An owner
//! with no resident checkpoint is still known as hybrid, and that survives
//! dumps.

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

type TierOwners = FxHashSet<ResidencyOwner>;

/// One current checkpoint record. Group completeness is resolved by the
/// publishing worker; readers need neither owner-layout joins nor publication
/// certificates.
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
    /// The owner's complete checkpoints: hash to one bit per storage tier.
    complete_checkpoints: FxHashMap<u64, u8>,
    /// Declared with a checkpoint alignment this table cannot use. The owner
    /// stays known as hybrid; its checkpoint events are not recorded.
    closed: bool,
    retired: bool,
    /// Mirror of `layout().complete_layout` that queries load without this lock.
    published_layout: Arc<AtomicBool>,
}

impl OwnerPresence {
    /// Call after any change to the declared alignment or retirement.
    fn publish_layout(&self) {
        self.published_layout
            .store(self.layout().complete_layout, Ordering::Relaxed);
    }

    fn layout(&self) -> CheckpointLayout {
        CheckpointLayout {
            complete_layout: !self.retired && !self.closed,
        }
    }

    /// Returns the checkpoint's new tier mask if the store changed it.
    fn store(&mut self, hash: u64, tier: StorageTier) -> Option<u8> {
        let bit = tier_bit(tier);
        let mask = self.complete_checkpoints.entry(hash).or_default();
        if *mask & bit != 0 {
            return None;
        }
        *mask |= bit;
        Some(*mask)
    }

    /// Returns the checkpoint's new tier mask if the removal changed it.
    fn remove(&mut self, hash: u64, tier: StorageTier) -> Option<u8> {
        let bit = tier_bit(tier);
        let mask = self.complete_checkpoints.get_mut(&hash)?;
        if *mask & bit == 0 {
            return None;
        }
        *mask &= !bit;
        let mask = *mask;
        if mask == 0 {
            self.complete_checkpoints.remove(&hash);
        }
        Some(mask)
    }
}

/// What a query needs to know about a hybrid owner besides its checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLayout {
    /// The owner's checkpoint events are recorded, so a missing checkpoint
    /// means the engine cannot resume there. False for a retired owner and for
    /// one that declared an unsupported alignment: it is known as hybrid and
    /// gets no checkpoint credit.
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
        // A new owner's checkpoint events are recorded from its first one.
        presence.publish_layout();
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

    /// Declare whether an owner publishes recurrent checkpoints, and at which
    /// alignment. A non-recurrent declaration retires a worker. Only an
    /// alignment of one block is supported: any other value keeps the owner
    /// known as hybrid with no usable checkpoint.
    pub fn declare_layout(&self, owner: ResidencyOwner, recurrent: bool, alignment_blocks: u32) {
        if !recurrent {
            if let ResidencyOwner::Worker(worker) = owner {
                self.remove_worker_rank(worker);
            }
            return;
        }
        let state = self.get_or_create(owner);
        let mut state = state.write();
        let closed = alignment_blocks > 1;
        if state.retired || state.closed == closed {
            return;
        }
        state.closed = closed;
        state.publish_layout();
        if closed {
            tracing::warn!(
                ?owner,
                alignment_blocks,
                "unsupported recurrent checkpoint alignment; the owner gets no checkpoint credit"
            );
            let previous = std::mem::take(&mut state.complete_checkpoints);
            self.publish_reset(owner, previous, &state);
        }
    }

    /// Tagged events are consumed even if unsupported; never send them to attention.
    pub fn apply(&self, event: &RouterEvent) -> bool {
        let Some(tag) = event.kv_cache_group else {
            return false;
        };
        if tag.kind != KvCacheGroupKind::RecurrentCheckpoint {
            return true;
        }
        let Ok(owner) = event.residency_owner() else {
            return true;
        };
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                // Registers the owner as hybrid even when the event carries no
                // checkpoint, which is how a publisher announces itself.
                let state = self.get_or_create(owner);
                let mut state = state.write();
                if state.retired
                    || state.closed
                    || tag.alignment_blocks.is_some_and(|alignment| alignment > 1)
                {
                    return true;
                }
                let updates = data
                    .blocks
                    .iter()
                    .filter_map(|block| {
                        let hash = block.block_hash.0;
                        state
                            .store(hash, event.storage_tier)
                            .map(|mask| (hash, mask))
                    })
                    .collect();
                self.checkpoints.publish(owner, updates);
            }
            KvCacheEventData::Removed(data) => {
                if let Some(state) = self.owner(owner) {
                    let mut state = state.write();
                    if state.retired {
                        return true;
                    }
                    let updates = data
                        .block_hashes
                        .iter()
                        .filter_map(|hash| {
                            state
                                .remove(hash.0, event.storage_tier)
                                .map(|mask| (hash.0, mask))
                        })
                        .collect();
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

    /// One store per owner and tier. An owner's home tier is dumped even when
    /// empty: that an owner is hybrid and holds no checkpoint must survive
    /// recovery. An owner closed by its declaration is left to be declared again.
    pub fn dump_events(&self) -> Vec<RouterEvent> {
        let registry = self.registry.load();
        let mut out = Vec::new();
        for (&owner, entry) in &registry.owners {
            let state = entry.state.read();
            if state.closed {
                continue;
            }
            let home_tier = match owner {
                ResidencyOwner::Worker(_) => StorageTier::Device,
                ResidencyOwner::CacheOwner(_) => StorageTier::HostPinned,
            };
            for tier in [
                StorageTier::Device,
                StorageTier::HostPinned,
                StorageTier::Disk,
                StorageTier::External,
            ] {
                if matches!(owner, ResidencyOwner::CacheOwner(_)) && tier == StorageTier::Device {
                    continue;
                }
                let blocks: Vec<_> = state
                    .complete_checkpoints
                    .iter()
                    .filter(|(_, mask)| **mask & tier_bit(tier) != 0)
                    .map(|(&hash, _)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(0),
                        mm_extra_info: None,
                    })
                    .collect();
                if blocks.is_empty() && tier != home_tier {
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
                    kind: KvCacheGroupKind::RecurrentCheckpoint,
                    alignment_blocks: Some(1),
                }));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::KvCacheRemoveData;

    fn checkpoint(worker: WorkerWithDpRank, tier: StorageTier, hash: u64) -> RouterEvent {
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
            kind: KvCacheGroupKind::RecurrentCheckpoint,
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
    fn an_owner_without_checkpoints_is_hybrid_and_survives_dump() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        // What a publisher sends at startup: a tagged store that names no checkpoint.
        let mut announce = checkpoint(worker, StorageTier::Device, 11);
        announce.event.data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: Vec::new(),
        });
        assert!(table.apply(&announce));
        assert!(table.is_hybrid(worker));
        assert!(
            table
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        assert!(restored.is_hybrid(worker));
        assert!(
            restored
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        assert!(!has_checkpoint(&restored, owner, 11, StorageTier::Device));
        restored.apply(&checkpoint(worker, StorageTier::Device, 11));
        assert!(has_checkpoint(&restored, owner, 11, StorageTier::Device));
    }

    #[test]
    fn tier_membership_and_removal_are_independent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let chain = [ExternalSequenceBlockHash(11)];
        let mut event = checkpoint(worker, StorageTier::HostPinned, 11);
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
        event.storage_tier = StorageTier::Device;
        table.apply(&event);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: chain.to_vec(),
        });
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
    }

    #[test]
    fn cache_clear_keeps_the_owner_hybrid_but_worker_retirement_removes_it() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, true, 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
        let clear = RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                dp_rank: 0,
                data: KvCacheEventData::Cleared,
            },
        );
        table.clear_event(&clear);
        assert!(table.is_hybrid(worker));
        assert!(
            table
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        table.remove_worker(7);
        assert!(!table.is_active());
    }

    #[test]
    fn unknown_group_kind_is_consumed_without_mutation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let mut event = checkpoint(worker, StorageTier::Device, 11);
        event.kv_cache_group.as_mut().unwrap().kind = KvCacheGroupKind::Unknown;
        assert!(table.apply(&event));
        assert!(!table.is_active());
    }

    #[test]
    fn read_view_exposes_each_tier_and_duplicate_events_are_idempotent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, true, 1);
        let mut declared = table.read_view();
        assert!(declared.track_owner(owner).unwrap().complete_layout);
        assert!(
            declared
                .owners_at(hash, StorageTier::Device)
                .next()
                .is_none()
        );

        let mut event = checkpoint(worker, StorageTier::Device, 11);
        table.apply(&event);
        table.apply(&event);
        table.apply(&checkpoint(worker, StorageTier::HostPinned, 11));
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
    fn declarations_close_reopen_and_retire_an_owner() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        // An owner known only from its events keeps them when it is declared.
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        table.declare_layout(owner, true, 1);
        table.declare_layout(owner, true, 1);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));

        table.declare_layout(owner, true, 2);
        let mut unsupported = table.read_view();
        assert!(!unsupported.track_owner(owner).unwrap().complete_layout);
        assert!(
            unsupported
                .owners_at(hash, StorageTier::Device)
                .next()
                .is_none()
        );
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(table.is_hybrid(worker));
        assert!(table.dump_events().is_empty());

        table.declare_layout(owner, true, 1);
        assert!(
            table
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));

        table.declare_layout(owner, false, 1);
        assert!(!table.is_hybrid(worker));
        assert!(!table.is_active());
        assert!(table.checkpoints.entries.is_empty());
    }

    #[test]
    fn read_view_tracks_only_requested_owners_from_its_registry_snapshot() {
        let table = KvCacheGroupResidency::new();
        let a = WorkerWithDpRank::new(7, 0);
        let b = WorkerWithDpRank::new(8, 0);
        let later = WorkerWithDpRank::new(9, 0);
        for worker in [a, b] {
            table.apply(&checkpoint(worker, StorageTier::Device, 11));
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
        table.apply(&checkpoint(later, StorageTier::Device, 11));
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
    }

    #[test]
    fn membership_survives_dump_and_clear_keeps_the_owner_hybrid() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, true, 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        table.apply(&checkpoint(worker, StorageTier::HostPinned, 22));
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
            view.owners_at(ExternalSequenceBlockHash(22), StorageTier::Device)
                .next()
                .is_none()
        );
        assert!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::HostPinned)
                .next()
                .is_none()
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
        let event = checkpoint(worker, StorageTier::HostPinned, 11);
        table.apply(&event);
        let tag = event.kv_cache_group;
        let mut event =
            RouterEvent::with_cache_owner(7, event.event, StorageTier::HostPinned, source);
        event.kv_cache_group = tag;
        table.apply(&event);
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
        table.declare_layout(owner, true, 2);
        table.apply(&checkpoint(declared, StorageTier::Device, 11));
        let mut event = checkpoint(tagged, StorageTier::Device, 11);
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

        table.declare_layout(owner, true, 1);
        table.apply(&checkpoint(declared, StorageTier::Device, 11));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn current_records_observe_remove_readd_and_reclaim_empty_entries() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        let mut view = table.read_view();
        view.track_owner(owner);
        table.clear_worker(worker);
        assert!(view.owners_at(hash, StorageTier::Device).next().is_none());
        assert!(table.checkpoints.entries.is_empty());
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
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
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
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
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
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
        table.apply(&checkpoint(worker, StorageTier::Device, 11));
        let old = table.owner(owner).unwrap();
        table.remove_worker_rank(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 22));
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
                    table.apply(&checkpoint(worker, StorageTier::Device, hash));
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
            table.apply(&checkpoint(worker, StorageTier::Device, 11));
        }
        table.apply(&checkpoint(first, StorageTier::HostPinned, 11));
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
                    table.apply(&checkpoint(worker, StorageTier::Device, hash));
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

    /// The engine's per-group events through the worker's normalizer into this
    /// table: one checkpoint once the last group arrives, gone when one leaves.
    #[test]
    fn vllm_group_events_reach_presence_as_one_checkpoint() {
        use dynamo_kv_events::zmq_wire::{RawKvEvent, ZmqEventNormalizer, decode_event_batch};
        use rmp_serde::from_slice;

        let store = |group: u32| -> RawKvEvent {
            let payload = serde_json::json!([0.0, [{
                "type": "BlockStored", "block_hashes": [333], "parent_block_hash": 222,
                "token_ids": [], "block_size": 576, "medium": "CPU", "group_idx": group,
                "kv_cache_spec_kind": "mamba"
            }], 0]);
            let wire = rmp_serde::to_vec_named(&payload).unwrap();
            decode_event_batch(&wire).unwrap().events.remove(0)
        };
        let remove = |group: u32| -> RawKvEvent {
            from_slice(
                &rmp_serde::to_vec_named(&serde_json::json!({
                    "type": "BlockRemoved", "block_hashes": [333], "medium": "CPU",
                    "group_idx": group
                }))
                .unwrap(),
            )
            .unwrap()
        };
        let worker = WorkerWithDpRank::new(7, 0);
        assert!(
            ZmqEventNormalizer::new(576)
                .preprocess(store(0), worker)
                .is_none(),
            "default publisher must not emit tagged events"
        );
        let mut normalizer = ZmqEventNormalizer::new(576).with_recurrent_groups(Some(vec![0, 1]));
        assert!(
            normalizer.normalize(store(0), 1, worker).is_none(),
            "one of two groups is not a checkpoint"
        );
        let stored = normalizer
            .normalize(store(1), 1, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        let KvCacheEventData::Stored(data) = &stored.event.data else {
            panic!("store");
        };
        assert_eq!(data.blocks.len(), 1);
        assert_eq!(data.blocks[0].block_hash, ExternalSequenceBlockHash(333));
        assert!(!stored.targets_primary().unwrap());
        let presence = KvCacheGroupResidency::new();
        presence.apply(&stored);
        let has_checkpoint = || {
            let mut view = presence.read_view();
            let owner = stored.residency_owner().unwrap();
            view.track_owner(owner);
            view.owners_at(ExternalSequenceBlockHash(333), StorageTier::HostPinned)
                .any(|known| known == owner)
        };
        assert!(has_checkpoint());
        let removed = normalizer
            .normalize(remove(0), 2, worker)
            .unwrap()
            .into_router_event()
            .unwrap();
        presence.apply(&removed);
        assert!(!has_checkpoint());
        assert!(
            normalizer.normalize(remove(1), 3, worker).is_none(),
            "the checkpoint was already gone"
        );
    }

    /// Routers used to keep one residency per recurrent group and call a
    /// checkpoint complete where every group was resident in one tier. Now the
    /// worker joins the groups and this table only records its events. Drive
    /// both with the same random stream of per-group engine events and require
    /// the same complete checkpoints after every event.
    #[test]
    fn joined_group_events_match_a_per_group_reference() {
        use dynamo_kv_events::zmq_wire::{
            BlockHashValue, KvCacheSpecKind, RawKvEvent, ZmqEventNormalizer,
        };

        const GROUPS: u32 = 3;
        const HASHES: u64 = 12;
        let tiers = [
            (None, StorageTier::Device),
            (Some("CPU"), StorageTier::HostPinned),
        ];
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let mut normalizer =
            ZmqEventNormalizer::new(4).with_recurrent_groups(Some((0..GROUPS).collect()));
        let table = KvCacheGroupResidency::new();
        // The per-group reference: (hash, tier) to the groups resident there.
        let mut reference: FxHashMap<(u64, StorageTier), u32> = FxHashMap::default();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let (mut published, mut complete_seen) = (0u32, 0u32);
        for step in 0..4_000u64 {
            let (medium, tier) = tiers[next(2) as usize];
            let group = next(u64::from(GROUPS)) as u32;
            let first = 1 + next(HASHES - 2);
            let hashes: Vec<u64> = (first..=first + next(3)).collect();
            let block_hashes = hashes
                .iter()
                .map(|&hash| BlockHashValue::Unsigned(hash))
                .collect();
            let raw = match next(400) {
                0 => {
                    reference.clear();
                    RawKvEvent::AllBlocksCleared { ownership: None }
                }
                1..=220 => {
                    for &hash in &hashes {
                        *reference.entry((hash, tier)).or_default() |= 1 << group;
                    }
                    RawKvEvent::BlockStored {
                        block_hashes,
                        parent_block_hash: None,
                        token_ids: Vec::new(),
                        block_size: 4,
                        medium: medium.map(str::to_string),
                        lora_name: None,
                        cache_namespace: None,
                        block_mm_infos: None,
                        is_eagle: None,
                        group_idx: Some(group),
                        kv_cache_spec_kind: Some(KvCacheSpecKind::Mamba),
                        kv_cache_spec_sliding_window: None,
                        locality: None,
                        ownership: None,
                        session_id: None,
                    }
                }
                _ => {
                    for &hash in &hashes {
                        if let Some(resident) = reference.get_mut(&(hash, tier)) {
                            *resident &= !(1 << group);
                        }
                    }
                    RawKvEvent::BlockRemoved {
                        block_hashes,
                        medium: medium.map(str::to_string),
                        group_idx: Some(group),
                        kv_cache_spec_kind: None,
                        kv_cache_spec_sliding_window: None,
                        locality: None,
                        ownership: None,
                    }
                }
            };
            if let Some(event) = normalizer.normalize(raw, step, worker) {
                published += 1;
                let event = event.into_router_event().unwrap();
                // As the indexer does: tagged events go to this table, and an
                // untagged clear drops the worker's checkpoints.
                if !table.apply(&event) {
                    table.clear_event(&event);
                }
            }
            for hash in 1..=HASHES {
                for (_, tier) in tiers {
                    let expected = reference.get(&(hash, tier)) == Some(&((1 << GROUPS) - 1));
                    complete_seen += u32::from(expected);
                    assert_eq!(
                        has_checkpoint(&table, owner, hash, tier),
                        expected,
                        "step {step}, hash {hash}, {tier:?}"
                    );
                }
            }
        }
        // The stream must have exercised both sides, and the join must have
        // published far fewer events than the engine reported.
        assert!(complete_seen > 1_000, "{complete_seen}");
        assert!(published > 100 && published < 3_000, "{published}");
    }
}
