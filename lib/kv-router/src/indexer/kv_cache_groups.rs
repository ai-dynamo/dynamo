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

use std::{
    cell::OnceCell,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;

const CHECKPOINT_SHARDS: usize = 256;

type CheckpointShardMap = FxHashMap<u64, Arc<CompleteMembership>>;
type CapturedCheckpointShards = [OnceCell<Arc<CheckpointShardMap>>; CHECKPOINT_SHARDS];

fn compare_owners(left: &ResidencyOwner, right: &ResidencyOwner) -> std::cmp::Ordering {
    match (left, right) {
        (ResidencyOwner::Worker(left), ResidencyOwner::Worker(right)) => left.cmp(right),
        (ResidencyOwner::CacheOwner(left), ResidencyOwner::CacheOwner(right)) => left.cmp(right),
        (ResidencyOwner::Worker(_), ResidencyOwner::CacheOwner(_)) => std::cmp::Ordering::Less,
        (ResidencyOwner::CacheOwner(_), ResidencyOwner::Worker(_)) => std::cmp::Ordering::Greater,
    }
}

/// Read-only complete membership for one hash and tier inside a scoped callback.
/// Facts still require the caller's captured owner layout/generation before credit.
/// The private slice borrows the immutable shard held by that callback.
#[derive(Clone, Copy)]
pub struct CheckpointMembershipView<'a> {
    owners: &'a [(ResidencyOwner, u64)],
}

impl CheckpointMembershipView<'_> {
    pub fn len(&self) -> usize {
        self.owners.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ResidencyOwner, u64)> + '_ {
        self.owners.iter().copied()
    }

    pub fn generation_for(&self, owner: ResidencyOwner) -> Option<u64> {
        self.owners
            .binary_search_by(|entry| compare_owners(&entry.0, &owner))
            .ok()
            .map(|index| self.owners[index].1)
    }
}

#[derive(Debug, Default, Clone)]
struct CompleteMembership {
    // Sorted by the exact logical owner, with one generation per owner/tier.
    owners: [Vec<(ResidencyOwner, u64)>; 4],
    revision: u64,
}

impl CompleteMembership {
    fn update(&mut self, owner: ResidencyOwner, generation: u64, mask: u8) {
        for (index, owners) in self.owners.iter_mut().enumerate() {
            let found = owners.binary_search_by(|entry| compare_owners(&entry.0, &owner));
            match (found, mask & (1 << index) != 0) {
                (Ok(position), true) => owners[position].1 = generation,
                (Err(position), true) => owners.insert(position, (owner, generation)),
                (Ok(position), false) if owners[position].1 == generation => {
                    owners.remove(position);
                }
                _ => {}
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.owners.iter().all(Vec::is_empty)
    }
}

#[derive(Debug, Default)]
struct CheckpointShard {
    entries: ArcSwap<CheckpointShardMap>,
    // Revisions only compare memberships for the same hash, which always maps
    // to this shard. Retaining the counter across deletion prevents ABA reuse.
    writer: Mutex<u64>,
}

#[derive(Debug)]
struct CheckpointMembershipIndex {
    shards: [CheckpointShard; CHECKPOINT_SHARDS],
}

impl Default for CheckpointMembershipIndex {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| CheckpointShard::default()),
        }
    }
}

impl CheckpointMembershipIndex {
    fn shard(hash: u64) -> usize {
        // Sequence hashes are already uniformly distributed. Folding higher bits
        // also keeps synthetic sequential hashes spread across the shards.
        (hash ^ (hash >> 32) ^ (hash >> 16) ^ (hash >> 8)) as usize & (CHECKPOINT_SHARDS - 1)
    }

    /// The caller holds the owner's write guard through publication. Shard
    /// writers serialize different owners and never hold two shard locks.
    fn publish(&self, owner: ResidencyOwner, updates: Vec<(u64, u64, u8)>) {
        let mut batches: FxHashMap<usize, Vec<(u64, u64, u8)>> = FxHashMap::default();
        for update in updates {
            batches
                .entry(Self::shard(update.0))
                .or_default()
                .push(update);
        }
        for (shard_index, updates) in batches {
            let shard = &self.shards[shard_index];
            let mut counter = shard.writer.lock();
            // Zero permanently disables live hints after counter exhaustion;
            // mutations still publish, and no revision can wrap or be reused.
            let revision = counter.checked_add(1).unwrap_or(0);
            if revision != 0 {
                *counter = revision;
            }
            let mut next = (*shard.entries.load_full()).clone();
            for (hash, generation, mask) in updates {
                let mut membership = next
                    .get(&hash)
                    .map(|entry| (**entry).clone())
                    .unwrap_or_default();
                membership.update(owner, generation, mask);
                membership.revision = revision;
                if membership.is_empty() {
                    next.remove(&hash);
                } else {
                    next.insert(hash, Arc::new(membership));
                }
            }
            shard.entries.store(Arc::new(next));
        }
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
    generation: u64,
    retired: bool,
    layout_snapshot: Arc<ArcSwap<CapturedLayout>>,
}

impl OwnerPresence {
    fn publish_layout(&self) {
        self.layout_snapshot.store(Arc::new(CapturedLayout {
            layout: self.layout(),
            generation: self.generation,
        }));
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

/// Recurrent checkpoints are separate from attention edges and scoped to their owner.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, Arc<RwLock<OwnerPresence>>>,
    layouts: FxHashMap<ResidencyOwner, Arc<ArcSwap<CapturedLayout>>>,
    cache_owners: FxHashMap<ResidencyOwnerKey, ResidencyOwner>,
}

/// Immutable owner layout captured for one query. Keep this token associated
/// with the same owner when validating membership or seeding legacy lookups.
#[derive(Debug, Clone, Copy)]
pub struct CapturedLayout {
    layout: CheckpointLayout,
    generation: u64,
}

impl CapturedLayout {
    pub fn layout(self) -> CheckpointLayout {
        self.layout
    }

    pub fn is_eligible_generation(self, generation: u64) -> bool {
        self.generation == generation && self.layout.complete_layout
    }
}

impl Default for CapturedLayout {
    fn default() -> Self {
        Self {
            layout: CheckpointLayout {
                alignment_blocks: 1,
                complete_layout: false,
            },
            generation: 0,
        }
    }
}

/// Candidate layouts are captured once. Legacy membership lookups retain shards
/// lazily; revision-fenced lookups only hold a shard during each call.
#[derive(Debug)]
pub struct CheckpointReadView {
    registry: Arc<OwnerRegistry>,
    layouts: FxHashMap<ResidencyOwner, Option<CapturedLayout>>,
    memberships: Arc<CheckpointMembershipIndex>,
    shards: OnceCell<Box<CapturedCheckpointShards>>,
}

impl CheckpointReadView {
    /// Reuses any prior legacy capture, including an unknown owner. Otherwise
    /// loads the registry's immutable layout without allocating a query entry.
    /// The caller must retain the result to preserve its first observation.
    pub fn capture_layout(&self, owner: ResidencyOwner) -> Option<CapturedLayout> {
        if let Some(&captured) = self.layouts.get(&owner) {
            return captured;
        }
        self.registry
            .layouts
            .get(&owner)
            .map(|snapshot| **snapshot.load())
    }

    /// Seeds a legacy lookup with the caller's first capture for this same
    /// owner. An existing capture, including `None`, is never replaced.
    pub fn remember_layout(&mut self, owner: ResidencyOwner, captured: Option<CapturedLayout>) {
        self.layouts.entry(owner).or_insert(captured);
    }

    pub fn track_owner(&mut self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        let registry = &self.registry;
        self.layouts
            .entry(owner)
            .or_insert_with(|| {
                registry
                    .layouts
                    .get(&owner)
                    .map(|snapshot| **snapshot.load())
            })
            .map(CapturedLayout::layout)
    }

    pub fn layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        self.layouts
            .get(&owner)
            .copied()
            .flatten()
            .map(|captured| captured.layout)
    }

    pub fn owner_for_key(&self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        self.registry.cache_owners.get(&key).copied()
    }

    pub fn track_cache_owner(&mut self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        let owner = self.owner_for_key(key)?;
        self.track_owner(owner);
        Some(owner)
    }

    #[inline]
    fn membership(&self, hash: ExternalSequenceBlockHash) -> Option<&CompleteMembership> {
        let index = CheckpointMembershipIndex::shard(hash.0);
        let shards = self
            .shards
            .get_or_init(|| Box::new(std::array::from_fn(|_| OnceCell::new())));
        let shard =
            shards[index].get_or_init(|| self.memberships.shards[index].entries.load_full());
        shard.get(&hash.0).map(Arc::as_ref)
    }

    /// Returns a live hint for any owner in this tier, without retaining a shard.
    /// This is not proof of owner or generation eligibility. A revision visitor
    /// must validate this hint and the owner's captured layout before credit.
    #[inline]
    pub fn checkpoint_hint_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> Option<u64> {
        let shard = self.memberships.shards[CheckpointMembershipIndex::shard(hash.0)]
            .entries
            .load();
        let membership = shard.get(&hash.0)?;
        (membership.revision != 0 && !membership.owners[tier_index(tier)].is_empty())
            .then_some(membership.revision)
    }

    /// Borrows indexed raw membership only for the exact nonzero hash revision.
    /// The view cannot outlive this call's immutable shard guard. A fact alone
    /// grants no credit: validate the owner's captured layout and generation.
    ///
    /// ```compile_fail
    /// use dynamo_kv_router::indexer::KvCacheGroupResidency;
    /// use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, StorageTier};
    /// let residency = KvCacheGroupResidency::new();
    /// let view = residency.read_view();
    /// let mut escaped = None;
    /// view.with_membership_at_revision(
    ///     ExternalSequenceBlockHash(1), StorageTier::Device, 1,
    ///     |membership| escaped = Some(membership),
    /// );
    /// ```
    pub fn with_membership_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl for<'a> FnMut(CheckpointMembershipView<'a>),
    ) {
        if revision == 0 {
            return;
        }
        let shard = self.memberships.shards[CheckpointMembershipIndex::shard(hash.0)]
            .entries
            .load();
        let Some(membership) = shard.get(&hash.0) else {
            return;
        };
        if membership.revision != revision {
            return;
        }
        visit(CheckpointMembershipView {
            owners: &membership.owners[tier_index(tier)],
        });
    }

    /// Visits raw membership facts only while this hash has the exact observed
    /// revision. A fact alone grants no credit: the caller must match the owner
    /// to its first captured layout and validate generation and completeness
    /// with `CapturedLayout::is_eligible_generation` before granting credit.
    /// The immutable shard guard is scoped to this call and is never cached.
    pub fn for_each_membership_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl FnMut(ResidencyOwner, u64),
    ) {
        self.with_membership_at_revision(hash, tier, revision, |membership| {
            for (owner, generation) in membership.iter() {
                visit(owner, generation);
            }
        });
    }

    /// Visits eligible owners only if this hash still has the observed revision.
    /// A store, removal, or lifecycle change after the hint cannot introduce
    /// credit. Eligibility uses layouts already captured by this view.
    pub fn for_each_owner_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl FnMut(ResidencyOwner),
    ) {
        self.for_each_membership_at_revision(hash, tier, revision, |owner, generation| {
            if self
                .layouts
                .get(&owner)
                .copied()
                .flatten()
                .is_some_and(|captured| captured.is_eligible_generation(generation))
            {
                visit(owner);
            }
        });
    }

    pub fn owners_at(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> impl Iterator<Item = ResidencyOwner> + '_ {
        self.membership(hash)
            .into_iter()
            .flat_map(move |membership| membership.owners[tier_index(tier)].iter())
            .filter_map(|&(owner, generation)| {
                let captured = self.layouts.get(&owner)?.as_ref()?;
                (captured.generation == generation && captured.layout.complete_layout)
                    .then_some(owner)
            })
    }
}

#[derive(Debug, Default)]
pub struct KvCacheGroupResidency {
    registry: ArcSwap<OwnerRegistry>,
    registry_writer: Mutex<()>,
    active: AtomicBool,
    memberships: Arc<CheckpointMembershipIndex>,
    next_generation: AtomicU64,
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
            memberships: self.memberships.clone(),
            shards: OnceCell::new(),
        }
    }

    fn owner(&self, owner: ResidencyOwner) -> Option<Arc<RwLock<OwnerPresence>>> {
        self.registry.load().owners.get(&owner).cloned()
    }

    fn fresh_generation(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn publish_reset(
        &self,
        owner: ResidencyOwner,
        previous_generation: u64,
        previous: FxHashMap<u64, u8>,
        state: &OwnerPresence,
    ) {
        let mut updates = Vec::with_capacity(previous.len() + state.complete_checkpoints.len());
        updates.extend(
            previous
                .into_keys()
                .map(|hash| (hash, previous_generation, 0)),
        );
        updates.extend(
            state
                .complete_checkpoints
                .iter()
                .map(|(&hash, &mask)| (hash, state.generation, mask)),
        );
        self.memberships.publish(owner, updates);
    }

    fn clear_owner(&self, owner: ResidencyOwner, state: &mut OwnerPresence) {
        let previous = std::mem::take(&mut state.complete_checkpoints);
        let previous_generation = state.generation;
        state.clear_checkpoints();
        state.generation = self.fresh_generation();
        state.publish_layout();
        self.publish_reset(owner, previous_generation, previous, state);
    }

    fn get_or_create(&self, owner: ResidencyOwner) -> Arc<RwLock<OwnerPresence>> {
        if let Some(state) = self.owner(owner) {
            return state;
        }
        let _writer = self.registry_writer.lock();
        let current = self.registry.load_full();
        if let Some(state) = current.owners.get(&owner) {
            return state.clone();
        }
        let mut next = (*current).clone();
        let state = OwnerPresence {
            generation: self.fresh_generation(),
            ..OwnerPresence::default()
        };
        state.publish_layout();
        next.layouts.insert(owner, state.layout_snapshot.clone());
        let state = Arc::new(RwLock::new(state));
        next.owners.insert(owner, state.clone());
        if matches!(owner, ResidencyOwner::CacheOwner(_)) {
            next.cache_owners.insert(owner.compact_key(), owner);
        }
        self.registry.store(Arc::new(next));
        self.active.store(true, Ordering::Relaxed);
        state
    }

    /// Install the complete recurrent layout before accepting routing credit.
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
        let previous_generation = state.generation;
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
            state.generation = self.fresh_generation();
            state.publish_layout();
            self.publish_reset(owner, previous_generation, previous, &state);
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
                let previous_generation = state.generation;
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
                if layout_changed {
                    state.generation = self.fresh_generation();
                }
                let mut updates = Vec::new();
                for block in &data.blocks {
                    let hash = block.block_hash.0;
                    let old = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    state.store(tag.group_idx, block.block_hash.0, event.storage_tier);
                    let new = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    if !layout_changed && old != new {
                        updates.push((hash, state.generation, new));
                    }
                }
                if let Some(previous) = previous {
                    state.publish_layout();
                    self.publish_reset(owner, previous_generation, previous, &state);
                } else {
                    self.memberships.publish(owner, updates);
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
                            updates.push((hash.0, state.generation, new));
                        }
                    }
                    self.memberships.publish(owner, updates);
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
        next.layouts.retain(
            |owner, _| !matches!(owner, ResidencyOwner::Worker(worker) if matches(*worker)),
        );
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

    fn owners_at_revision(
        view: &CheckpointReadView,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
    ) -> Vec<ResidencyOwner> {
        let mut owners = Vec::new();
        view.for_each_owner_at_revision(hash, tier, revision, |owner| owners.push(owner));
        owners
    }

    #[test]
    fn indexed_membership_sorts_exact_owner_identity_and_dp_rank() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };
        use IdentitySource::{DefaultDerived, Explicit};

        let cache_owner = |semantics_source, routing_source, cluster, slot, slot_source| {
            ResidencyOwner::cache_owner(CacheOwnerId::new(
                PoolId::new(
                    IndexerDomainId::new(
                        CacheSemanticsId::new([1; 16], semantics_source),
                        RoutingScopeId::new([2; 16], routing_source),
                    ),
                    ClusterId::new(cluster),
                ),
                StableDpSlotId::new([slot; 16], slot_source),
            ))
        };
        let rank0 = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let rank2 = ResidencyOwner::worker(WorkerWithDpRank::new(7, 2));
        let later_worker = ResidencyOwner::worker(WorkerWithDpRank::new(u64::MAX, 0));
        let semantics_derived = cache_owner(DefaultDerived, Explicit, 3, 4, Explicit);
        let routing_derived = cache_owner(Explicit, DefaultDerived, 3, 4, Explicit);
        let slot_derived = cache_owner(Explicit, Explicit, 3, 4, DefaultDerived);
        let explicit = cache_owner(Explicit, Explicit, 3, 4, Explicit);
        let later_slot = cache_owner(Explicit, Explicit, 3, 5, Explicit);
        let later_cluster = cache_owner(Explicit, Explicit, 4, 0, Explicit);
        let expected = [
            rank0,
            rank2,
            later_worker,
            semantics_derived,
            routing_derived,
            slot_derived,
            explicit,
            later_slot,
            later_cluster,
        ];
        let insertion_order = [
            later_cluster,
            rank2,
            explicit,
            semantics_derived,
            later_worker,
            later_slot,
            rank0,
            routing_derived,
            slot_derived,
        ];
        let mut membership = CompleteMembership::default();
        for (index, owner) in insertion_order.into_iter().enumerate() {
            membership.update(owner, index as u64 + 10, 0b1111);
        }
        for owners in &membership.owners {
            let indexed = CheckpointMembershipView { owners };
            assert_eq!(indexed.len(), expected.len());
            assert!(!indexed.is_empty());
            assert_eq!(
                indexed.iter().map(|(owner, _)| owner).collect::<Vec<_>>(),
                expected,
            );
            for (index, owner) in insertion_order.into_iter().enumerate() {
                assert_eq!(indexed.generation_for(owner), Some(index as u64 + 10));
            }
            assert_eq!(
                indexed.generation_for(ResidencyOwner::worker(WorkerWithDpRank::new(7, 1))),
                None,
            );
            assert_eq!(
                indexed.generation_for(cache_owner(Explicit, Explicit, 3, 3, Explicit)),
                None,
            );
        }
    }

    #[test]
    fn indexed_membership_removal_requires_the_matching_generation() {
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let middle = ResidencyOwner::worker(WorkerWithDpRank::new(8, 0));
        let last = ResidencyOwner::worker(WorkerWithDpRank::new(9, 0));
        let mut membership = CompleteMembership::default();
        for owner in [last, first, middle] {
            membership.update(owner, 11, 0b0011);
        }
        membership.update(middle, 12, 0b0011);
        membership.update(middle, 11, 0);
        for owners in &membership.owners[..2] {
            let indexed = CheckpointMembershipView { owners };
            assert_eq!(indexed.len(), 3);
            assert_eq!(indexed.generation_for(middle), Some(12));
        }
        membership.update(middle, 12, 0);
        for owners in &membership.owners[..2] {
            let indexed = CheckpointMembershipView { owners };
            assert_eq!(indexed.len(), 2);
            assert_eq!(indexed.generation_for(middle), None);
            assert_eq!(indexed.generation_for(first), Some(11));
            assert_eq!(indexed.generation_for(last), Some(11));
        }
        membership.update(middle, 13, 0b0001);
        assert_eq!(
            membership.owners[0]
                .iter()
                .map(|&(owner, _)| owner)
                .collect::<Vec<_>>(),
            [first, middle, last],
        );
        let device = CheckpointMembershipView {
            owners: &membership.owners[0],
        };
        assert_eq!(device.generation_for(middle), Some(13));
        assert!(
            CheckpointMembershipView {
                owners: &membership.owners[1]
            }
            .generation_for(middle)
            .is_none()
        );
    }

    #[test]
    fn indexed_revision_view_fences_updates_and_preserves_cached_snapshots() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(9, 2);
        let first_owner = ResidencyOwner::worker(first);
        let second_owner = ResidencyOwner::worker(second);
        let hash = ExternalSequenceBlockHash(11);
        for worker in [second, first] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let mut view = table.read_view();
        for owner in [first_owner, second_owner] {
            view.track_owner(owner);
        }
        let original = view
            .owners_at(hash, StorageTier::Device)
            .collect::<Vec<_>>();
        assert_eq!(original, [first_owner, second_owner]);
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut first_generation = None;
        view.with_membership_at_revision(hash, StorageTier::Device, revision, |membership| {
            assert_eq!(membership.len(), 2);
            first_generation = membership.generation_for(first_owner);
            assert_eq!(
                membership
                    .iter()
                    .map(|(owner, _)| owner)
                    .collect::<Vec<_>>(),
                original,
            );
        });
        assert!(first_generation.is_some());
        view.with_membership_at_revision(hash, StorageTier::HostPinned, revision, |membership| {
            assert!(membership.is_empty());
            assert_eq!(membership.generation_for(first_owner), None);
        });
        for invalid in [0, revision + 1] {
            view.with_membership_at_revision(hash, StorageTier::Device, invalid, |_| {
                panic!("invalid revision must not expose membership")
            });
        }
        view.with_membership_at_revision(
            ExternalSequenceBlockHash(12),
            StorageTier::Device,
            revision,
            |_| panic!("unknown hash must not expose membership"),
        );

        table.clear_worker(first);
        view.with_membership_at_revision(hash, StorageTier::Device, revision, |_| {
            panic!("old revision must not survive a clear")
        });
        let cleared = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        view.with_membership_at_revision(hash, StorageTier::Device, cleared, |membership| {
            assert_eq!(membership.len(), 1);
            assert_eq!(membership.generation_for(first_owner), None);
            assert!(membership.generation_for(second_owner).is_some());
        });
        table.apply(&checkpoint(first, StorageTier::Device, 0, hash.0, Some(1)));
        let readded = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(readded, cleared);
        view.with_membership_at_revision(hash, StorageTier::Device, cleared, |_| {
            panic!("clear/readd must not reuse a revision")
        });
        view.with_membership_at_revision(hash, StorageTier::Device, readded, |membership| {
            assert_eq!(membership.len(), 2);
            assert_ne!(membership.generation_for(first_owner), first_generation);
        });
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            original,
        );
    }

    #[test]
    fn uncached_layouts_seed_legacy_lookups_without_recapturing_new_lifecycles() {
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        for lifecycle in 0..3 {
            let table = KvCacheGroupResidency::new();
            table.declare_layout(owner, &[0], 1);
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
            let mut view = table.read_view();
            let captured = view.capture_layout(owner).unwrap();
            assert!(captured.layout().complete_layout);
            assert!(view.layouts.is_empty());
            match lifecycle {
                0 => table.clear_worker(worker),
                1 => table.declare_layout(owner, &[3], 1),
                _ => {
                    table.remove_worker_rank(worker);
                    table.declare_layout(owner, &[0], 1);
                }
            }
            let group = if lifecycle == 1 { 3 } else { 0 };
            table.apply(&checkpoint(
                worker,
                StorageTier::Device,
                group,
                hash.0,
                Some(1),
            ));
            let fresh = table.read_view().capture_layout(owner).unwrap();
            assert_ne!(captured.generation, fresh.generation);
            assert!(!captured.is_eligible_generation(fresh.generation));

            view.remember_layout(owner, Some(captured));
            view.remember_layout(owner, Some(fresh));
            assert_eq!(
                view.capture_layout(owner).unwrap().generation,
                captured.generation
            );
            assert_eq!(view.track_owner(owner), Some(captured.layout()));
            assert!(view.owners_at(hash, StorageTier::Device).next().is_none());
            let revision = view
                .checkpoint_hint_revision(hash, StorageTier::Device)
                .unwrap();
            assert!(owners_at_revision(&view, hash, StorageTier::Device, revision).is_empty());
        }
    }

    #[test]
    fn layout_capture_reuses_prior_legacy_some_and_none() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, &[0], 1);
        let mut legacy = table.read_view();
        legacy.track_owner(owner);
        let original = legacy.capture_layout(owner).unwrap();
        table.clear_worker(worker);
        let current = table.read_view().capture_layout(owner).unwrap();
        assert_ne!(original.generation, current.generation);
        assert_eq!(
            legacy.capture_layout(owner).unwrap().generation,
            original.generation
        );
        legacy.remember_layout(owner, Some(current));
        assert_eq!(
            legacy.capture_layout(owner).unwrap().generation,
            original.generation
        );

        let unknown_worker = WorkerWithDpRank::new(8, 0);
        let unknown_owner = ResidencyOwner::worker(unknown_worker);
        let mut unknown = table.read_view();
        let captured = unknown.capture_layout(unknown_owner);
        assert!(captured.is_none());
        assert!(unknown.layouts.is_empty());
        table.declare_layout(unknown_owner, &[0], 1);
        table.apply(&checkpoint(
            unknown_worker,
            StorageTier::Device,
            0,
            11,
            Some(1),
        ));
        unknown.remember_layout(unknown_owner, captured);
        unknown.remember_layout(
            unknown_owner,
            table.read_view().capture_layout(unknown_owner),
        );
        assert!(unknown.layouts.contains_key(&unknown_owner));
        assert!(unknown.capture_layout(unknown_owner).is_none());
        assert!(unknown.track_owner(unknown_owner).is_none());
        let hash = ExternalSequenceBlockHash(11);
        let revision = unknown
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&unknown, hash, StorageTier::Device, revision).is_empty());
    }

    #[test]
    fn raw_revision_facts_need_captured_owner_and_complete_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        let unknown = table.read_view();
        let unknown_capture = unknown.capture_layout(owner);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let revision = unknown
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut facts = Vec::new();
        unknown.for_each_membership_at_revision(
            hash,
            StorageTier::Device,
            revision,
            |fact_owner, generation| facts.push((fact_owner, generation)),
        );
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].0, owner);
        assert!(!unknown_capture.is_some_and(|token| token.is_eligible_generation(facts[0].1)));
        assert!(owners_at_revision(&unknown, hash, StorageTier::Device, revision).is_empty());
        assert!(unknown.layouts.is_empty());
        assert!(unknown.shards.get().is_none());

        let current = table.read_view();
        let captured = current.capture_layout(owner).unwrap();
        assert!(captured.is_eligible_generation(facts[0].1));
        let incomplete = WorkerWithDpRank::new(8, 0);
        table.apply(&checkpoint(
            incomplete,
            StorageTier::Device,
            0,
            hash.0,
            None,
        ));
        let incomplete = table
            .read_view()
            .capture_layout(ResidencyOwner::worker(incomplete))
            .unwrap();
        assert!(!incomplete.layout().complete_layout);
        assert!(!incomplete.is_eligible_generation(incomplete.generation));

        table.clear_worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let revision = current
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut credit = Vec::new();
        current.for_each_membership_at_revision(
            hash,
            StorageTier::Device,
            revision,
            |fact_owner, generation| {
                if fact_owner == owner && captured.is_eligible_generation(generation) {
                    credit.push(fact_owner);
                }
            },
        );
        assert!(credit.is_empty());
    }

    #[test]
    fn live_hints_fence_changed_memberships_without_caching_shards() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let other_worker = WorkerWithDpRank::new(8, 0);
        let owner = ResidencyOwner::worker(worker);
        let other = ResidencyOwner::worker(other_worker);
        let hash = ExternalSequenceBlockHash(11);
        let sibling = ExternalSequenceBlockHash(266);
        assert_eq!(
            CheckpointMembershipIndex::shard(hash.0),
            CheckpointMembershipIndex::shard(sibling.0)
        );
        table.declare_layout(owner, &[0], 1);
        table.declare_layout(other, &[0], 1);
        let mut view = table.read_view();
        view.track_owner(owner);
        view.track_owner(other);
        let generation = view.layouts[&other].unwrap().generation;
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );

        let mut event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        let KvCacheEventData::Stored(data) = &mut event.event.data else {
            unreachable!()
        };
        data.blocks.push(KvCacheStoredBlockData {
            block_hash: sibling,
            tokens_hash: LocalBlockHash(0),
            mm_extra_info: None,
        });
        table.apply(&event);
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_eq!(
            view.checkpoint_hint_revision(sibling, StorageTier::Device),
            Some(revision)
        );
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::HostPinned),
            None
        );
        assert_eq!(
            view.checkpoint_hint_revision(ExternalSequenceBlockHash(12), StorageTier::Device),
            None
        );
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, revision),
            [owner]
        );
        assert!(owners_at_revision(&view, hash, StorageTier::HostPinned, revision).is_empty());
        table.apply(&event);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            Some(revision)
        );

        // A later owner cannot acquire credit using an earlier tree hint.
        table.apply(&checkpoint(
            other_worker,
            StorageTier::Device,
            0,
            hash.0,
            Some(1),
        ));
        let changed = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(changed, revision);
        assert!(owners_at_revision(&view, hash, StorageTier::Device, revision).is_empty());
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, changed),
            [owner, other]
        );
        // Publishing the same shard does not invalidate unrelated hashes.
        assert_eq!(
            view.checkpoint_hint_revision(sibling, StorageTier::Device),
            Some(revision)
        );
        let mut current = table.read_view();
        current.track_owner(other);
        assert_eq!(current.layouts[&other].unwrap().generation, generation);
        assert!(view.shards.get().is_none());
    }

    #[test]
    fn live_revision_rejects_remove_readd_aba_without_changing_layout_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let mut view = table.read_view();
        view.track_owner(owner);
        let generation = view.layouts[&owner].unwrap().generation;
        let before = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut remove = event.clone();
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        assert!(owners_at_revision(&view, hash, StorageTier::Device, before).is_empty());
        table.apply(&event);
        let after = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(before, after);
        assert!(owners_at_revision(&view, hash, StorageTier::Device, before).is_empty());
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, after),
            [owner]
        );
        let mut current = table.read_view();
        current.track_owner(owner);
        assert_eq!(current.layouts[&owner].unwrap().generation, generation);
        assert!(view.shards.get().is_none());
    }

    #[test]
    fn live_revision_still_requires_tracked_current_layout_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let mut old = table.read_view();
        let revision = old
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&old, hash, StorageTier::Device, revision).is_empty());
        old.track_owner(owner);

        table.declare_layout(owner, &[3], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 3, hash.0, Some(1)));
        let revision = old
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&old, hash, StorageTier::Device, revision).is_empty());
        let mut current = table.read_view();
        current.track_owner(owner);
        assert_eq!(
            owners_at_revision(&current, hash, StorageTier::Device, revision),
            [owner]
        );
        assert!(old.shards.get().is_none());
        assert!(current.shards.get().is_none());
    }

    #[test]
    fn membership_revision_exhaustion_disables_live_credit_without_wrapping() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        let shard = &table.memberships.shards[CheckpointMembershipIndex::shard(hash.0)];
        *shard.writer.lock() = u64::MAX - 1;
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let mut view = table.read_view();
        view.track_owner(owner);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            Some(u64::MAX)
        );
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, u64::MAX),
            [owner]
        );
        table.apply(&checkpoint(
            worker,
            StorageTier::HostPinned,
            0,
            hash.0,
            Some(1),
        ));
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        assert!(owners_at_revision(&view, hash, StorageTier::Device, u64::MAX).is_empty());
        assert!(owners_at_revision(&view, hash, StorageTier::Device, 0).is_empty());
        let mut remove = event.clone();
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        table.apply(&event);
        assert_eq!(*shard.writer.lock(), u64::MAX);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        // Cached lower-tier lookups continue to expose the actual membership.
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
    }

    #[test]
    fn short_query_snapshots_only_visited_shards_without_copying_checkpoints() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 1, Some(1));
        let KvCacheEventData::Stored(data) = &mut event.event.data else {
            unreachable!()
        };
        data.blocks = (1..=1024)
            .map(|hash| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(hash),
                tokens_hash: LocalBlockHash(0),
                mm_extra_info: None,
            })
            .collect();
        table.apply(&event);

        let mut view = table.read_view();
        view.track_owner(owner);
        assert!(view.shards.get().is_none());
        assert_eq!(view.layouts.len(), 1);
        for hash in 1..=8 {
            assert_eq!(
                view.owners_at(ExternalSequenceBlockHash(hash), StorageTier::Device)
                    .collect::<Vec<_>>(),
                [owner]
            );
        }
        assert_eq!(
            view.shards
                .get()
                .unwrap()
                .iter()
                .filter(|shard| shard.get().is_some())
                .count(),
            8
        );

        // Duplicate events do not publish new shards. Removing one checkpoint
        // only replaces its shard; all unrelated shard snapshots stay shared.
        let before: Vec<_> = table
            .memberships
            .shards
            .iter()
            .map(|shard| shard.entries.load_full())
            .collect();
        table.apply(&event);
        assert!(
            before
                .iter()
                .zip(&table.memberships.shards)
                .all(|(old, shard)| Arc::ptr_eq(old, &shard.entries.load_full()))
        );
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1)],
        });
        table.apply(&event);
        // Legacy lookups keep the original per-shard snapshot across writes.
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(1), StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
        assert_eq!(
            before
                .iter()
                .zip(&table.memberships.shards)
                .filter(|(old, shard)| !Arc::ptr_eq(old, &shard.entries.load_full()))
                .count(),
            1
        );
        let mut removed = table.read_view();
        removed.track_owner(owner);
        assert!(
            removed
                .owners_at(ExternalSequenceBlockHash(1), StorageTier::Device)
                .next()
                .is_none()
        );
        assert_eq!(
            removed
                .owners_at(ExternalSequenceBlockHash(2), StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
    }

    #[test]
    fn lazy_snapshots_reject_membership_from_new_owner_lifecycles() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));

        for lifecycle in 0..3 {
            let mut old = table.read_view();
            old.track_owner(owner);
            let old_generation = old.layouts[&owner].unwrap().generation;
            match lifecycle {
                0 => table.clear_worker(worker),
                1 => table.declare_layout(owner, &[0, 1], 1),
                _ => {
                    table.remove_worker_rank(worker);
                    table.declare_layout(owner, &[0, 1], 1);
                }
            }
            let groups = if lifecycle == 0 { 1 } else { 2 };
            for group in 0..groups {
                table.apply(&checkpoint(
                    worker,
                    StorageTier::Device,
                    group,
                    hash.0,
                    Some(groups),
                ));
            }
            let mut fresh = table.read_view();
            fresh.track_owner(owner);
            assert!(fresh.layouts[&owner].unwrap().generation > old_generation);
            assert!(old.owners_at(hash, StorageTier::Device).next().is_none());
            assert_eq!(
                fresh
                    .owners_at(hash, StorageTier::Device)
                    .collect::<Vec<_>>(),
                [owner]
            );
        }
    }

    #[test]
    fn concurrent_owner_batches_preserve_all_complete_memberships() {
        let table = KvCacheGroupResidency::new();
        std::thread::scope(|scope| {
            for id in 1..=8 {
                let table = &table;
                scope.spawn(move || {
                    let worker = WorkerWithDpRank::new(id, 0);
                    for group in [0, 1] {
                        let mut event = checkpoint(worker, StorageTier::Device, group, 1, Some(2));
                        let KvCacheEventData::Stored(data) = &mut event.event.data else {
                            unreachable!()
                        };
                        data.blocks = (1..=128)
                            .map(|hash| KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(hash),
                                tokens_hash: LocalBlockHash(0),
                                mm_extra_info: None,
                            })
                            .collect();
                        table.apply(&event);
                    }
                });
            }
        });
        let mut view = table.read_view();
        for id in 1..=8 {
            view.track_owner(ResidencyOwner::worker(WorkerWithDpRank::new(id, 0)));
        }
        for hash in 1..=128 {
            let mut owners = view
                .owners_at(ExternalSequenceBlockHash(hash), StorageTier::Device)
                .collect::<Vec<_>>();
            owners.sort_by_key(|owner| match owner {
                ResidencyOwner::Worker(worker) => worker.worker_id,
                _ => unreachable!(),
            });
            assert_eq!(
                owners,
                (1..=8)
                    .map(|id| ResidencyOwner::worker(WorkerWithDpRank::new(id, 0)))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn read_view_does_not_wait_for_owner_or_membership_writer_locks() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let mut view = table.read_view();
        let state = table.owner(owner).unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let owner_writer = state.write();
            let shard_writer = table.memberships.shards[CheckpointMembershipIndex::shard(11)]
                .writer
                .lock();
            scope.spawn(move || {
                assert!(view.track_owner(owner).unwrap().complete_layout);
                send.send(
                    view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            });
            let result = receive.recv_timeout(std::time::Duration::from_secs(5));
            // Drop guards before asserting so a regression fails instead of
            // leaving a blocked scoped thread behind during panic unwinding.
            drop(shard_writer);
            drop(owner_writer);
            assert_eq!(result.unwrap(), [owner]);
        });
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
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
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
}
