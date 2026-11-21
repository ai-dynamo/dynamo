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

use std::ops::Deref;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use arc_swap::{ArcSwap, Guard};
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

/// An independently owned set with the same equal-set shortcut as CRTC's
/// `WorkerSet`: compare length and a 64-bit XOR fingerprint. Unequal sets can
/// collide (about 2^-64 per comparison), which can overestimate a worker's
/// resumable checkpoint. This deliberately adopts CRTC's probabilistic
/// equality contract; the members themselves are never shared between hashes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CheckpointWorkerSet {
    workers: FxHashSet<WorkerWithDpRank>,
    fingerprint: u64,
}

impl CheckpointWorkerSet {
    fn insert(&mut self, worker: WorkerWithDpRank) -> bool {
        let inserted = self.workers.insert(worker);
        if inserted {
            self.fingerprint ^= checkpoint_worker_fingerprint(worker);
        }
        inserted
    }

    fn remove(&mut self, worker: &WorkerWithDpRank) -> bool {
        let removed = self.workers.remove(worker);
        if removed {
            self.fingerprint ^= checkpoint_worker_fingerprint(*worker);
        }
        removed
    }

    /// Constant-time comparison, subject to the documented collision risk.
    #[cfg(test)]
    fn same_members(&self, other: &Self) -> bool {
        self.workers.len() == other.workers.len() && self.fingerprint == other.fingerprint
    }

    /// Size and fingerprint. Equal signatures mean equal members, subject to
    /// the documented collision risk.
    pub(crate) fn signature(&self) -> (usize, u64) {
        (self.workers.len(), self.fingerprint)
    }
}

impl Deref for CheckpointWorkerSet {
    type Target = FxHashSet<WorkerWithDpRank>;

    fn deref(&self) -> &Self::Target {
        &self.workers
    }
}

#[cfg(test)]
impl FromIterator<WorkerWithDpRank> for CheckpointWorkerSet {
    fn from_iter<T: IntoIterator<Item = WorkerWithDpRank>>(workers: T) -> Self {
        let mut set = Self::default();
        for worker in workers {
            set.insert(worker);
        }
        set
    }
}

// Keep the same worker-id/DP-rank mixing as
// concurrent_radix_tree_compressed::worker_set::mix_worker, without exposing or
// changing that tree's private worker-set implementation.
#[inline]
fn checkpoint_worker_fingerprint(worker: WorkerWithDpRank) -> u64 {
    #[inline]
    fn mix(value: u64) -> u64 {
        let mut value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    mix(mix(worker.worker_id) ^ u64::from(worker.dp_rank))
}

/// One current checkpoint record. Group completeness is resolved on the write
/// side; readers need neither owner-layout joins nor publication certificates.
#[derive(Debug, Default)]
struct CheckpointPresence {
    owners: [TierOwners; 4],
    // Each hash owns its workers, even when another hash has identical members.
    device_workers: CheckpointWorkerSet,
}

/// Owners held on each lower tier, summed over all records. Zero is exact,
/// so a query skips a lower tier that has no checkpoint; a nonzero value may
/// be briefly stale-high. Device is not counted: every Device checkpoint
/// would write here, and each worker already has its own Device count.
///
/// The counters sit on their own cache lines, away from the directory fields
/// that every lookup reads.
#[derive(Debug, Default)]
#[repr(align(128))]
struct LowerTierMembers([AtomicUsize; 3]);

impl LowerTierMembers {
    fn joined(&self, tier_index: usize) {
        if let Some(count) = tier_index.checked_sub(1) {
            self.0[count].fetch_add(1, Ordering::Relaxed);
        }
    }

    fn left(&self, tier_index: usize) {
        if let Some(count) = tier_index.checked_sub(1) {
            self.0[count].fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Whether any owner holds a checkpoint on this tier. Device is unknown
    /// here and reported as held.
    fn any(&self, tier_index: usize) -> bool {
        tier_index
            .checked_sub(1)
            .is_none_or(|count| self.0[count].load(Ordering::Relaxed) > 0)
    }
}

/// How one record update changed the owner's Device worker-set membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeviceMembership {
    Unchanged,
    Joined,
    Left,
}

impl CheckpointPresence {
    #[cfg(test)]
    fn update(&mut self, owner: ResidencyOwner, mask: u8) -> DeviceMembership {
        self.update_counted(owner, mask, &Default::default())
    }

    /// `members` counts, per lower tier, the owners held across all records.
    /// It is raised before a record that gained an owner becomes visible and
    /// lowered after one lost it, under the record's shard guard.
    fn update_counted(
        &mut self,
        owner: ResidencyOwner,
        mask: u8,
        members: &LowerTierMembers,
    ) -> DeviceMembership {
        if self
            .owners
            .iter()
            .enumerate()
            .all(|(index, owners)| owners.contains(&owner) == (mask & (1 << index) != 0))
        {
            return DeviceMembership::Unchanged;
        }
        for (index, owners) in self.owners.iter_mut().enumerate() {
            if mask & (1 << index) != 0 {
                if owners.insert(owner) {
                    members.joined(index);
                }
            } else if owners.remove(&owner) {
                members.left(index);
            }
        }
        if let Some(worker) = owner.as_worker() {
            if mask & 1 != 0 {
                if self.device_workers.insert(worker) {
                    return DeviceMembership::Joined;
                }
            } else if self.device_workers.remove(&worker) {
                return DeviceMembership::Left;
            }
        }
        DeviceMembership::Unchanged
    }

    fn is_empty(&self) -> bool {
        self.owners.iter().all(FxHashSet::is_empty)
    }
}

/// Slots in the Device filter: a power of two, sixteen bytes each.
const DEVICE_FILTER_SLOTS: usize = 1 << 18;
const DEVICE_FILTER_SHIFT: u32 = u64::BITS - DEVICE_FILTER_SLOTS.trailing_zeros();

/// The tag of a slot that names no single hash: empty, or shared by the
/// Device sets of several hashes. A query then reads the record.
const DEVICE_FILTER_ANY: u32 = 0;

/// One Device filter slot. Queries load `workers` and `tag` and nothing else.
#[derive(Default)]
struct DeviceFilterSlot {
    /// Union of the filter bits of the workers in every Device set that maps
    /// here. Bits are cleared only when the last such set empties.
    workers: AtomicU64,
    /// Fingerprint of the hash whose Device set is the only one that maps
    /// here, so a query for another hash skips the record read. It is
    /// [`DEVICE_FILTER_ANY`] from the second set on until the slot empties:
    /// the slot does not know which hash is left when one of several leaves.
    tag: AtomicU32,
    /// Number of nonempty Device sets that map here. The mutex serializes
    /// this slot's writers, which hold it for a few instructions.
    records: Mutex<u16>,
}

// Queries load one slot per block, so a slot stays at two words.
const _: () = assert!(std::mem::size_of::<DeviceFilterSlot>() == 16);

impl DeviceFilterSlot {
    /// A worker joined the Device set of the hash with `fingerprint`, which
    /// maps here; `first` when the set was empty before.
    fn joined(&self, bit: u64, first: bool, fingerprint: u32) {
        // A set that was already nonempty keeps this slot from being cleared
        // while its shard guard is held, so a bit seen here stays.
        if !first && self.workers.load(Ordering::Relaxed) & bit != 0 {
            return;
        }
        let mut records = self.records.lock();
        self.workers.fetch_or(bit, Ordering::Release);
        if first {
            // A full counter stays full, and its slot is never cleared.
            *records = records.saturating_add(1);
            let tag = if *records == 1 {
                fingerprint
            } else {
                DEVICE_FILTER_ANY
            };
            self.tag.store(tag, Ordering::Release);
        }
    }

    /// A Device set that maps here lost its last worker.
    fn emptied(&self) {
        let mut records = self.records.lock();
        if *records == u16::MAX {
            return;
        }
        *records = records.saturating_sub(1);
        if *records == 0 {
            self.workers.store(0, Ordering::Release);
            self.tag.store(DEVICE_FILTER_ANY, Ordering::Release);
        }
    }
}

/// A fixed table that tells a query, without touching the directory, that no
/// record at a hash can hold the workers it is looking for.
///
/// Every registered worker has one of 64 filter bits. A slot's bits are a
/// superset of the workers held by the Device sets that map to it, so a clear
/// answer is exact and a set bit only means "read the record". A slot that
/// holds the Device set of one hash also carries that hash's fingerprint, so
/// a query for another hash that maps there is answered without a read.
/// Writers update a slot under the record's shard guard, before the change is
/// visible.
struct DeviceFilter {
    slots: Box<[DeviceFilterSlot]>,
}

impl std::fmt::Debug for DeviceFilter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceFilter")
            .field("slots", &self.slots.len())
            .finish()
    }
}

impl DeviceFilter {
    fn new() -> Self {
        Self {
            slots: (0..DEVICE_FILTER_SLOTS)
                .map(|_| DeviceFilterSlot::default())
                .collect(),
        }
    }

    /// One multiply keeps the slot independent of the directory's buckets.
    fn index(hash: u64) -> usize {
        (hash.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> DEVICE_FILTER_SHIFT) as usize
    }

    fn slot(&self, hash: u64) -> &DeviceFilterSlot {
        &self.slots[Self::index(hash)]
    }

    /// A nonzero tag from product bits that the slot index does not use.
    fn fingerprint(hash: u64) -> u32 {
        (hash.wrapping_mul(0x9e37_79b9_7f4a_7c15) as u32) | 1
    }

    /// Whether a Device set at `hash` may hold a worker with one of `bits`.
    fn may_hold(&self, hash: u64, bits: u64) -> bool {
        let slot = self.slot(hash);
        if slot.workers.load(Ordering::Acquire) & bits == 0 {
            return false;
        }
        // A tag that names another hash means this hash has no Device set
        // here, or one that is not visible yet.
        let tag = slot.tag.load(Ordering::Acquire);
        tag == DEVICE_FILTER_ANY || tag == Self::fingerprint(hash)
    }
}

/// The directory's shard guard protects record reads, publication, and removal.
/// A box keeps each hash-table value small without a per-record lock or shared
/// reference count. Queries release the guard before matching or callbacks.
#[derive(Debug, Default)]
struct CheckpointDirectory {
    entries: DashMap<u64, Box<CheckpointPresence>, FxBuildHasher>,
    /// Built on the first Device join; a deployment without one never pays
    /// for it.
    filter: OnceLock<DeviceFilter>,
    lower_tier_members: LowerTierMembers,
    /// Checkpoints that routing decisions predict, by worker.
    predicted: DashMap<WorkerWithDpRank, Vec<PredictedCheckpoint>, FxBuildHasher>,
    /// How long a prediction lives, in milliseconds. Zero turns prediction off.
    prediction_ttl_ms: AtomicU64,
}

/// A checkpoint the router expects a worker to hold soon: the last full block
/// of a prompt it was just routed. It stands in for the worker's checkpoint
/// event until that arrives or the prediction expires.
#[derive(Clone, Copy, Debug)]
struct PredictedCheckpoint {
    position: usize,
    hash: ExternalSequenceBlockHash,
    expires: Instant,
}

/// Predictions kept per worker. Only a repeat or a continuation of a recent
/// prompt can match one, so a few are enough.
const PREDICTED_PER_WORKER: usize = 8;

impl CheckpointDirectory {
    fn predict(&self, worker: WorkerWithDpRank, position: usize, hash: ExternalSequenceBlockHash) {
        let ttl = self.prediction_ttl_ms.load(Ordering::Relaxed);
        if ttl == 0 || position == 0 {
            return;
        }
        let now = Instant::now();
        let mut points = self.predicted.entry(worker).or_default();
        points.retain(|point| {
            point.expires > now && !(point.position == position && point.hash == hash)
        });
        if points.len() >= PREDICTED_PER_WORKER {
            points.remove(0);
        }
        points.push(PredictedCheckpoint {
            position,
            hash,
            expires: now + Duration::from_millis(ttl),
        });
    }

    /// The deepest live prediction for `worker` at or below `end` whose block
    /// is the request's block at that position.
    fn predicted_point(
        &self,
        worker: WorkerWithDpRank,
        end: usize,
        request_hash_at: impl Fn(usize) -> Option<ExternalSequenceBlockHash>,
    ) -> Option<(usize, ExternalSequenceBlockHash)> {
        if self.prediction_ttl_ms.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let points = self.predicted.get(&worker)?;
        let now = Instant::now();
        points
            .iter()
            .filter(|point| {
                point.position <= end
                    && point.expires > now
                    && request_hash_at(point.position) == Some(point.hash)
            })
            .map(|point| (point.position, point.hash))
            .max_by_key(|&(position, _)| position)
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Directory-only publication for tests that do not read the owner count.
    #[cfg(test)]
    fn publish(&self, owner: ResidencyOwner, updates: Vec<(u64, u8)>) {
        self.publish_counted(owner, updates, &AtomicUsize::new(0), 1);
    }

    /// `device_checkpoints` counts the records whose Device worker set holds
    /// this owner. It is raised before any record can gain the owner and
    /// lowered only after records lose it, so a reader that sees zero knows
    /// no record credits the owner; a nonzero value may be briefly stale-high.
    /// `filter_bit` is the owner's bit in the Device filter.
    fn publish_counted(
        &self,
        owner: ResidencyOwner,
        updates: Vec<(u64, u8)>,
        device_checkpoints: &AtomicUsize,
        filter_bit: u64,
    ) {
        let may_join = if owner.as_worker().is_some() {
            updates.iter().filter(|(_, mask)| mask & 1 != 0).count()
        } else {
            0
        };
        if may_join > 0 {
            device_checkpoints.fetch_add(may_join, Ordering::Relaxed);
        }
        let mut joined = 0;
        let mut left = 0;
        for (hash, mask) in updates {
            // Update and remove under one entry guard so concurrent writers
            // cannot publish into a detached record. No owner lock is acquired
            // while holding this guard, and each iteration releases it.
            let change = match self.entries.entry(hash) {
                Entry::Occupied(mut entry) => {
                    let (change, empty) = {
                        let record = entry.get_mut();
                        let change = record.update_counted(owner, mask, &self.lower_tier_members);
                        let holders = record.device_workers.len();
                        self.track_filter(hash, filter_bit, change, holders);
                        (change, record.is_empty())
                    };
                    if empty {
                        entry.remove();
                    }
                    change
                }
                Entry::Vacant(entry) if mask != 0 => {
                    let mut record = CheckpointPresence::default();
                    let change = record.update_counted(owner, mask, &self.lower_tier_members);
                    let holders = record.device_workers.len();
                    self.track_filter(hash, filter_bit, change, holders);
                    entry.insert(Box::new(record));
                    change
                }
                Entry::Vacant(_) => DeviceMembership::Unchanged,
            };
            match change {
                DeviceMembership::Joined => joined += 1,
                DeviceMembership::Left => left += 1,
                DeviceMembership::Unchanged => {}
            }
        }
        // Every join was pre-counted, so `may_join >= joined`.
        let surplus = may_join - joined + left;
        if surplus > 0 {
            device_checkpoints.fetch_sub(surplus, Ordering::Relaxed);
        }
    }

    /// Mirror one record's Device membership change into the filter. Runs
    /// under the record's shard guard, with `holders` the set's new size.
    fn track_filter(&self, hash: u64, bit: u64, change: DeviceMembership, holders: usize) {
        match change {
            DeviceMembership::Joined => self
                .filter
                .get_or_init(DeviceFilter::new)
                .slot(hash)
                .joined(bit, holders == 1, DeviceFilter::fingerprint(hash)),
            DeviceMembership::Left if holders == 0 => {
                if let Some(filter) = self.filter.get() {
                    filter.slot(hash).emptied();
                }
            }
            DeviceMembership::Left | DeviceMembership::Unchanged => {}
        }
    }

    /// Run `visit` on this hash's nonempty Device worker set under the shard
    /// read guard. `bits` are the filter bits of the workers the caller is
    /// looking for; the directory is not touched when the filter rules all of
    /// them out. The visitor must only touch query-local state.
    fn visit_device_workers(
        &self,
        hash: ExternalSequenceBlockHash,
        bits: u64,
        visit: impl FnOnce(&CheckpointWorkerSet),
    ) {
        if !self
            .filter
            .get()
            .is_some_and(|filter| filter.may_hold(hash.0, bits))
        {
            return;
        }
        if let Some(record) = self.entries.get(&hash.0)
            && !record.device_workers.is_empty()
        {
            visit(&record.device_workers);
        }
    }

    /// Run `visit` on this hash's nonempty owner set of `tier` under the shard
    /// read guard. Nothing is copied out of the record; the visitor must be
    /// short and touch only query-local state.
    fn visit_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&TierOwners),
    ) {
        if let Some(record) = self.entries.get(&hash.0) {
            let owners = &record.owners[tier_index(tier)];
            if !owners.is_empty() {
                visit(owners);
            }
        }
    }

    fn tier_has_checkpoints(&self, tier: StorageTier) -> bool {
        self.lower_tier_members.any(tier_index(tier))
    }

    #[cfg(test)]
    fn with_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&TierOwners),
    ) -> bool {
        let owners = {
            let Some(record) = self.entries.get(&hash.0) else {
                return false;
            };
            let owners = &record.owners[tier_index(tier)];
            if owners.is_empty() {
                return false;
            }
            owners.clone()
        };
        visit(&owners);
        true
    }
}

/// The Device checkpoint records, for an evaluator that settles workers
/// against them from its own view of the table.
#[derive(Clone, Copy)]
pub(crate) struct DeviceRecords<'a>(&'a CheckpointDirectory);

impl DeviceRecords<'_> {
    /// See [`KvCacheGroupResidency::visit_checkpoint_workers`].
    pub(crate) fn visit(
        &self,
        hash: ExternalSequenceBlockHash,
        bits: u64,
        visit: impl FnOnce(&CheckpointWorkerSet),
    ) {
        self.0.visit_device_workers(hash, bits, visit);
    }
}

#[cfg(test)]
fn compare_owners(left: &ResidencyOwner, right: &ResidencyOwner) -> std::cmp::Ordering {
    match (left, right) {
        (ResidencyOwner::Worker(left), ResidencyOwner::Worker(right)) => left.cmp(right),
        (ResidencyOwner::CacheOwner(left), ResidencyOwner::CacheOwner(right)) => left.cmp(right),
        (ResidencyOwner::Worker(_), ResidencyOwner::CacheOwner(_)) => std::cmp::Ordering::Less,
        (ResidencyOwner::CacheOwner(_), ResidencyOwner::Worker(_)) => std::cmp::Ordering::Greater,
    }
}

pub(crate) fn tier_index(tier: StorageTier) -> usize {
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
    /// Shared with the registry snapshot so queries can read it without
    /// taking this owner's lock. See [`CheckpointDirectory::publish_counted`].
    device_checkpoints: Arc<AtomicUsize>,
    /// This owner's bit in the Device filter; zero for a cache owner, which
    /// never joins a Device set.
    filter_bit: u64,
    /// Blocks the attention match must extend past a checkpoint for the
    /// engine to resume there (EAGLE/MTP trailing-block drop); 0 otherwise.
    eagle_margin_blocks: u32,
    /// Mirror of `eagle_margin_blocks` that queries load without this lock.
    published_margin: Arc<AtomicU32>,
}

impl OwnerPresence {
    /// Call after any change to the group set, the expected count or retirement.
    fn publish_layout(&self) {
        let layout = self.layout();
        self.published_layout
            .store(layout.complete_layout, Ordering::Relaxed);
        self.published_margin
            .store(layout.eagle_margin_blocks, Ordering::Relaxed);
    }

    fn layout(&self) -> CheckpointLayout {
        CheckpointLayout {
            complete_layout: !self.retired
                && !self.groups.is_empty()
                && self.expected_group_count == Some(self.groups.len()),
            eagle_margin_blocks: self.eagle_margin_blocks,
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
    /// Blocks the attention match must extend past a checkpoint for the
    /// owner's engine to resume there. Zero unless the engine drops the
    /// trailing prefix-cache block for EAGLE/MTP.
    pub eagle_margin_blocks: u32,
}

/// Only registration topology is shared as a snapshot. Checkpoint data lives in
/// independent mutable records and has the same best-effort freshness as KV facts.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, OwnerEntry>,
    cache_owners: FxHashMap<ResidencyOwnerKey, ResidencyOwner>,
    /// What queries read about each registered worker, keyed like `owners`.
    workers: FxHashMap<WorkerWithDpRank, WorkerSummary>,
    /// Workers registered so far; the next one takes filter bit `n % 64`.
    filter_bits_assigned: u32,
}

#[derive(Debug, Clone)]
struct OwnerEntry {
    state: Arc<RwLock<OwnerPresence>>,
    complete_layout: Arc<AtomicBool>,
    eagle_margin: Arc<AtomicU32>,
}

/// What a query reads about one registered worker without taking its lock.
#[derive(Debug, Clone)]
struct WorkerSummary {
    /// See [`CheckpointDirectory::publish_counted`].
    device_checkpoints: Arc<AtomicUsize>,
    /// The worker's bit in the Device filter.
    filter_bit: u64,
    /// See [`CheckpointLayout::eagle_margin_blocks`].
    eagle_margin: Arc<AtomicU32>,
}

#[derive(Debug)]
pub(crate) struct KnownWorkerRead {
    registry: Guard<Arc<OwnerRegistry>>,
}

impl KnownWorkerRead {
    #[cfg(test)]
    pub(crate) fn is_known_worker(&self, worker: WorkerWithDpRank) -> bool {
        self.registry
            .owners
            .contains_key(&ResidencyOwner::worker(worker))
    }

    /// How many checkpoint records currently credit this worker on Device,
    /// and the worker's Device filter bit; `None` for a worker that was
    /// unknown when this view was captured. A zero count is exact; a nonzero
    /// one may be briefly stale-high.
    pub(crate) fn device_summary(&self, worker: WorkerWithDpRank) -> Option<(usize, u64)> {
        self.registry.workers.get(&worker).map(|summary| {
            (
                summary.device_checkpoints.load(Ordering::Relaxed),
                summary.filter_bit,
            )
        })
    }

    /// Blocks the attention match must extend past a checkpoint for this
    /// worker's engine to resume there; zero for an unknown worker.
    pub(crate) fn device_margin(&self, worker: WorkerWithDpRank) -> u32 {
        self.registry
            .workers
            .get(&worker)
            .map_or(0, |summary| summary.eagle_margin.load(Ordering::Relaxed))
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
        let entry = self.registry.owners.get(&owner)?;
        // A registered owner remains known to this query even if retirement
        // happens before its first layout read. Its retired layout is incomplete,
        // so absent checkpoints receive zero credit rather than dense KV credit.
        // The owner lock is not taken: writers hold it across republication.
        Some(CheckpointLayout {
            complete_layout: entry.complete_layout.load(Ordering::Relaxed),
            eagle_margin_blocks: entry.eagle_margin.load(Ordering::Relaxed),
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

    /// Visit the owners that hold a complete checkpoint at `hash` on `tier`
    /// now, in place: the visitor runs under the directory shard read guard,
    /// so it must be short and touch only query-local state. Separate calls
    /// may see newer versions.
    pub(crate) fn visit_owners(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        visit: impl FnOnce(&FxHashSet<ResidencyOwner>),
    ) {
        self.checkpoints.visit_owners(hash, tier, visit);
    }

    /// Whether any owner holds a checkpoint on a lower `tier`. `false` is
    /// exact; Device always reports `true`.
    pub(crate) fn tier_has_checkpoints(&self, tier: StorageTier) -> bool {
        self.checkpoints.tier_has_checkpoints(tier)
    }

    /// How many checkpoint records credit this worker on Device and the
    /// worker's Device filter bit; `None` for a worker that was unknown when
    /// this view was captured. See [`KnownWorkerRead::device_summary`].
    pub(crate) fn device_summary(&self, worker: WorkerWithDpRank) -> Option<(usize, u64)> {
        self.registry.workers.get(&worker).map(|summary| {
            (
                summary.device_checkpoints.load(Ordering::Relaxed),
                summary.filter_bit,
            )
        })
    }

    pub(crate) fn device_records(&self) -> DeviceRecords<'_> {
        DeviceRecords(&self.checkpoints)
    }

    /// The deepest live predicted checkpoint of `worker` at or below `end`
    /// that sits on the request's own chain, as `(position, hash)`.
    pub(crate) fn predicted_checkpoint(
        &self,
        worker: WorkerWithDpRank,
        end: usize,
        request_hash_at: impl Fn(usize) -> Option<ExternalSequenceBlockHash>,
    ) -> Option<(usize, ExternalSequenceBlockHash)> {
        self.checkpoints
            .predicted_point(worker, end, request_hash_at)
    }

    /// The callback receives a copy of the current complete owners. All guards
    /// are released before invoking the callback; separate calls may see newer
    /// versions.
    #[cfg(test)]
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
    #[cfg(test)]
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

/// A query keeps its own worker set. Matching size and fingerprint reuse that
/// query-local set; otherwise return an independent copy of this hash's workers.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CheckpointWorkers {
    Unchanged,
    Changed(CheckpointWorkerSet),
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

    pub(crate) fn known_worker_read(&self) -> KnownWorkerRead {
        KnownWorkerRead {
            registry: self.registry.load(),
        }
    }

    /// Read this hash's complete Device workers. Each record owns
    /// its set independently. Compare size and fingerprint with the query's
    /// current set under the shard read guard, copying when they differ. This
    /// uses the probabilistic equality contract of `CheckpointWorkerSet`.
    /// The returned set owns its storage; matching runs after all guards drop.
    #[cfg(test)]
    pub(crate) fn checkpoint_workers_since(
        &self,
        hash: ExternalSequenceBlockHash,
        previous: Option<&CheckpointWorkerSet>,
    ) -> Option<CheckpointWorkers> {
        let record = self.checkpoints.entries.get(&hash.0)?;
        let workers = &record.device_workers;
        if workers.is_empty() {
            return None;
        }
        if let Some(previous) = previous
            && previous.same_members(workers)
        {
            debug_assert_eq!(previous.workers, workers.workers);
            return Some(CheckpointWorkers::Unchanged);
        }
        Some(CheckpointWorkers::Changed(workers.clone()))
    }

    /// Visit this hash's complete Device workers in place.
    /// `bits` are the Device filter bits of the workers the caller is looking
    /// for; a record that cannot hold any of them is skipped without a
    /// directory read. The visitor runs under the directory shard read guard,
    /// so it must be short and touch only query-local state.
    pub(crate) fn visit_checkpoint_workers(
        &self,
        hash: ExternalSequenceBlockHash,
        bits: u64,
        visit: impl FnOnce(&CheckpointWorkerSet),
    ) {
        self.checkpoints.visit_device_workers(hash, bits, visit);
    }

    pub(crate) fn device_records(&self) -> DeviceRecords<'_> {
        DeviceRecords(&self.checkpoints)
    }

    #[cfg(test)]
    fn checkpoint_workers(
        &self,
        hash: ExternalSequenceBlockHash,
    ) -> Option<FxHashSet<WorkerWithDpRank>> {
        match self.checkpoint_workers_since(hash, None)? {
            CheckpointWorkers::Changed(workers) => Some(workers.workers),
            CheckpointWorkers::Unchanged => unreachable!("no prior query set was supplied"),
        }
    }

    #[cfg(test)]
    fn with_checkpoint_workers(
        &self,
        hash: ExternalSequenceBlockHash,
        visit: impl FnOnce(&FxHashSet<WorkerWithDpRank>),
    ) -> bool {
        let Some(workers) = self.checkpoint_workers(hash) else {
            return false;
        };
        visit(&workers);
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
        self.checkpoints.publish_counted(
            owner,
            updates,
            &state.device_checkpoints,
            state.filter_bit,
        );
    }

    fn clear_owner(&self, owner: ResidencyOwner, state: &mut OwnerPresence) {
        if let Some(worker) = owner.as_worker() {
            self.checkpoints.predicted.remove(&worker);
        }
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
        let mut presence = OwnerPresence::default();
        if let Some(worker) = owner.as_worker() {
            presence.filter_bit = 1u64 << (next.filter_bits_assigned % u64::BITS);
            next.filter_bits_assigned = next.filter_bits_assigned.wrapping_add(1);
            next.workers.insert(
                worker,
                WorkerSummary {
                    device_checkpoints: presence.device_checkpoints.clone(),
                    filter_bit: presence.filter_bit,
                    eagle_margin: presence.published_margin.clone(),
                },
            );
        }
        let complete_layout = presence.published_layout.clone();
        let eagle_margin = presence.published_margin.clone();
        let state = Arc::new(RwLock::new(presence));
        next.owners.insert(
            owner,
            OwnerEntry {
                state: state.clone(),
                complete_layout,
                eagle_margin,
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
    /// `eagle_margin_blocks` is the attention margin the owner's engine needs
    /// past a checkpoint (see [`CheckpointLayout::eagle_margin_blocks`]).
    pub fn declare_layout(
        &self,
        owner: ResidencyOwner,
        groups: &[u32],
        alignment_blocks: u32,
        eagle_margin_blocks: u32,
    ) {
        if alignment_blocks > 1 && !groups.is_empty() {
            self.close_layout(owner, alignment_blocks);
            return;
        }
        self.install_layout(owner, groups, eagle_margin_blocks);
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

    fn install_layout(&self, owner: ResidencyOwner, groups: &[u32], eagle_margin_blocks: u32) {
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
        state.eagle_margin_blocks = eagle_margin_blocks;
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
                // The margin only changes how queries read checkpoints, so it
                // needs no checkpoint rebuild; `publish_layout` below exposes it.
                // A tag without the field keeps the declared margin.
                if let Some(margin) = tag.eagle_margin_blocks {
                    state.eagle_margin_blocks = margin;
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
                    self.checkpoints.publish_counted(
                        owner,
                        updates,
                        &state.device_checkpoints,
                        state.filter_bit,
                    );
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
                    self.checkpoints.publish_counted(
                        owner,
                        updates,
                        &state.device_checkpoints,
                        state.filter_bit,
                    );
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
        let owners = &next.owners;
        next.workers
            .retain(|worker, _| owners.contains_key(&ResidencyOwner::worker(*worker)));
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
                        eagle_margin_blocks: None,
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

    /// Let a routing decision predict a checkpoint for `ttl`, the lifetime of
    /// the predict-on-route overlay. `None` turns prediction off, the default.
    pub fn set_prediction_ttl(&self, ttl: Option<Duration>) {
        let millis = ttl.map_or(0, |ttl| ttl.as_millis().min(u128::from(u64::MAX)) as u64);
        self.checkpoints
            .prediction_ttl_ms
            .store(millis, Ordering::Relaxed);
    }

    /// A hybrid worker that was just routed a prompt of `position` full blocks
    /// ending at `hash` is expected to keep a checkpoint there.
    pub fn predict_checkpoint(
        &self,
        worker: WorkerWithDpRank,
        position: usize,
        hash: ExternalSequenceBlockHash,
    ) {
        if self.is_hybrid(worker) {
            self.checkpoints.predict(worker, position, hash);
        }
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
            eagle_margin_blocks: None,
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
        table.declare_layout(owner, &[0, 1, 2], 1, 0);
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
        table.declare_layout(ResidencyOwner::worker(worker), &[0], 1, 0);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn cache_clear_preserves_layout_but_worker_retirement_removes_it() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.declare_layout(ResidencyOwner::worker(worker), &[0, 1], 1, 0);
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
        table.declare_layout(owner, &[0, 1], 1, 0);
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

        table.declare_layout(owner, &[0, 1], 1, 0);
        table.declare_layout(owner, &[0, 1], 1, 0);
        let mut declared = table.read_view();
        declared.track_owner(owner);
        assert_eq!(
            declared
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        table.declare_layout(owner, &[0, 1, 2], 1, 0);
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

        table.declare_layout(owner, &[0, 1, 2], 2, 0);
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
        table.declare_layout(owner, &[0, 1], 1, 0);
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
                eagle_margin_blocks: None,
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
        table.declare_layout(owner, &[0], 2, 0);
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
        assert!(table.checkpoints.is_empty());

        table.declare_layout(owner, &[0], 1, 0);
        table.apply(&checkpoint(declared, StorageTier::Device, 0, 11, Some(1)));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn checkpoint_fingerprint_tracks_insert_remove_and_duplicate_updates() {
        let mut workers = CheckpointWorkerSet::default();
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..10_000 {
            let worker = WorkerWithDpRank::new(rng.u64(..32), rng.u32(..4));
            if rng.bool() {
                workers.insert(worker);
                let fingerprint = workers.fingerprint;
                assert!(!workers.insert(worker));
                assert_eq!(workers.fingerprint, fingerprint);
            } else {
                workers.remove(&worker);
                let fingerprint = workers.fingerprint;
                assert!(!workers.remove(&worker));
                assert_eq!(workers.fingerprint, fingerprint);
            }
            let rebuilt = workers
                .iter()
                .copied()
                .map(checkpoint_worker_fingerprint)
                .fold(0, |fingerprint, member| fingerprint ^ member);
            assert_eq!(workers.fingerprint, rebuilt);
        }
        let members = workers.iter().copied().collect::<Vec<_>>();
        for worker in members {
            assert!(workers.remove(&worker));
        }
        assert!(workers.is_empty());
        assert_eq!(workers.fingerprint, 0);
    }

    #[test]
    fn checkpoint_fingerprint_follows_tiers_and_owner_removal() {
        let first = WorkerWithDpRank::new(7, 0);
        let other_rank = WorkerWithDpRank::new(7, 1);
        let first_owner = ResidencyOwner::worker(first);
        let other_owner = ResidencyOwner::worker(other_rank);
        let mut record = CheckpointPresence::default();
        record.update(first_owner, 1);
        record.update(other_owner, 1);
        let both = CheckpointWorkerSet::from_iter([other_rank, first]);
        assert_eq!(record.device_workers, both);

        // An idempotent publish and a non-device tier change must preserve the
        // device fingerprint.
        record.update(first_owner, 1);
        record.update(first_owner, 3);
        assert_eq!(record.device_workers, both);
        record.update(first_owner, 1);
        assert_eq!(record.device_workers, both);
        record.update(other_owner, 2);
        assert_eq!(
            record.device_workers,
            CheckpointWorkerSet::from_iter([first])
        );
        record.update(first_owner, 0);
        record.update(other_owner, 0);
        assert!(record.is_empty());
        assert_eq!(record.device_workers, CheckpointWorkerSet::default());
    }

    #[test]
    fn checkpoint_worker_changes_distinguish_equal_counts_and_dp_ranks() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let other_rank = WorkerWithDpRank::new(7, 1);
        let hash = ExternalSequenceBlockHash(11);
        let previous = CheckpointWorkerSet::from_iter([first]);
        assert_eq!(table.checkpoint_workers_since(hash, Some(&previous)), None);

        table.apply(&checkpoint(first, StorageTier::Device, 0, hash.0, Some(1)));
        assert_eq!(
            table.checkpoint_workers_since(hash, Some(&previous)),
            Some(CheckpointWorkers::Unchanged)
        );
        let same_size_different_members = CheckpointWorkerSet::from_iter([other_rank]);
        assert_eq!(
            table.checkpoint_workers_since(hash, Some(&same_size_different_members)),
            Some(CheckpointWorkers::Changed(previous.clone()))
        );

        table.clear_worker(first);
        assert_eq!(table.checkpoint_workers_since(hash, Some(&previous)), None);
        table.apply(&checkpoint(
            other_rank,
            StorageTier::Device,
            0,
            hash.0,
            Some(1),
        ));
        assert_eq!(
            table.checkpoint_workers_since(hash, Some(&previous)),
            Some(CheckpointWorkers::Changed(same_size_different_members))
        );
        assert_eq!(previous.workers, FxHashSet::from_iter([first]));
    }

    #[test]
    fn equal_checkpoint_records_and_query_sets_mutate_independently() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let first_hash = ExternalSequenceBlockHash(11);
        let second_hash = ExternalSequenceBlockHash(22);
        for worker in [first, second] {
            table.checkpoints.publish(
                ResidencyOwner::worker(worker),
                vec![(first_hash.0, 1), (second_hash.0, 1)],
            );
        }
        let mut captured =
            CheckpointWorkerSet::from_iter(table.checkpoint_workers(first_hash).unwrap());
        assert_eq!(
            table.checkpoint_workers_since(second_hash, Some(&captured)),
            Some(CheckpointWorkers::Unchanged)
        );

        table
            .checkpoints
            .publish(ResidencyOwner::worker(second), vec![(first_hash.0, 0)]);
        assert_eq!(
            table.checkpoint_workers_since(first_hash, Some(&captured)),
            Some(CheckpointWorkers::Changed(CheckpointWorkerSet::from_iter(
                [first]
            )))
        );
        assert_eq!(
            table.checkpoint_workers_since(second_hash, Some(&captured)),
            Some(CheckpointWorkers::Unchanged)
        );

        // Query-local mutation cannot alter either record's independently owned set.
        captured.remove(&first);
        assert_eq!(
            table.checkpoint_workers(first_hash).unwrap(),
            FxHashSet::from_iter([first])
        );
        assert_eq!(
            table.checkpoint_workers(second_hash).unwrap(),
            FxHashSet::from_iter([first, second])
        );
    }

    #[test]
    fn device_filter_is_a_superset_and_clears_when_the_slot_empties() {
        let directory = CheckpointDirectory::default();
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let second = ResidencyOwner::worker(WorkerWithDpRank::new(8, 0));
        let count = AtomicUsize::new(0);
        let may_hold = |hash, bits| {
            directory
                .filter
                .get()
                .is_some_and(|filter| filter.may_hold(hash, bits))
        };
        // No Device set anywhere: nothing can hold anyone.
        assert!(!may_hold(11, u64::MAX));
        directory.publish_counted(first, vec![(11, 1)], &count, 0b01);
        assert!(may_hold(11, 0b01));
        assert!(!may_hold(11, 0b10));
        directory.publish_counted(second, vec![(11, 1)], &count, 0b10);
        assert!(may_hold(11, 0b10));
        // A departed worker's bit may linger, but a holder's bit never drops.
        directory.publish_counted(first, vec![(11, 0)], &count, 0b01);
        assert!(may_hold(11, 0b10));
        // Other tiers never enter the filter.
        assert_ne!(DeviceFilter::index(11), DeviceFilter::index(12));
        directory.publish_counted(first, vec![(12, 2)], &count, 0b01);
        assert!(!may_hold(12, u64::MAX));
        // The last Device set leaving the slot clears every bit.
        directory.publish_counted(second, vec![(11, 0)], &count, 0b10);
        assert!(!may_hold(11, u64::MAX));
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn device_filter_rules_out_another_hash_in_a_slot_that_names_one() {
        let directory = CheckpointDirectory::default();
        let worker = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let count = AtomicUsize::new(0);
        // Two hashes that share a filter slot but not a fingerprint.
        let first_hash = 1u64;
        let second_hash = (2u64..)
            .find(|&hash| {
                DeviceFilter::index(hash) == DeviceFilter::index(first_hash)
                    && DeviceFilter::fingerprint(hash) != DeviceFilter::fingerprint(first_hash)
            })
            .unwrap();
        let may_hold = |hash| directory.filter.get().unwrap().may_hold(hash, u64::MAX);
        directory.publish_counted(worker, vec![(first_hash, 1)], &count, 0b01);
        // The slot names the first hash, so the second is ruled out unread.
        assert!(may_hold(first_hash));
        assert!(!may_hold(second_hash));
        // With both sets in the slot it names neither.
        directory.publish_counted(worker, vec![(second_hash, 1)], &count, 0b01);
        assert!(may_hold(first_hash));
        assert!(may_hold(second_hash));
        // The slot cannot tell which hash is left, so it keeps answering "read".
        directory.publish_counted(worker, vec![(first_hash, 0)], &count, 0b01);
        assert!(may_hold(second_hash));
        assert!(may_hold(first_hash));
        // Once the slot empties, it names the next hash again.
        directory.publish_counted(worker, vec![(second_hash, 0)], &count, 0b01);
        assert!(!may_hold(second_hash));
        directory.publish_counted(worker, vec![(second_hash, 1)], &count, 0b01);
        assert!(may_hold(second_hash));
        assert!(!may_hold(first_hash));
        directory.publish_counted(worker, vec![(second_hash, 0)], &count, 0b01);
        assert!(directory.is_empty());
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn device_filter_keeps_a_held_bit_through_concurrent_slot_clears() {
        let directory = CheckpointDirectory::default();
        // Two hashes that share a filter slot but not a record.
        let first_hash = 1u64;
        let second_hash = (2u64..)
            .find(|&hash| DeviceFilter::index(hash) == DeviceFilter::index(first_hash))
            .unwrap();
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let second = ResidencyOwner::worker(WorkerWithDpRank::new(8, 0));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let count = AtomicUsize::new(0);
                for _ in 0..50_000 {
                    directory.publish_counted(first, vec![(first_hash, 1)], &count, 0b01);
                    directory.publish_counted(first, vec![(first_hash, 0)], &count, 0b01);
                }
            });
            scope.spawn(|| {
                let count = AtomicUsize::new(0);
                for _ in 0..50_000 {
                    directory.publish_counted(second, vec![(second_hash, 1)], &count, 0b10);
                    // The record holds the worker, so its bit must be visible.
                    assert!(directory.filter.get().unwrap().may_hold(second_hash, 0b10));
                    directory.publish_counted(second, vec![(second_hash, 0)], &count, 0b10);
                }
            });
        });
        assert!(directory.is_empty());
        let filter = directory.filter.get().unwrap();
        assert!(!filter.may_hold(first_hash, u64::MAX));
        assert_eq!(*filter.slot(first_hash).records.lock(), 0);
    }

    #[test]
    fn checkpoint_removal_preserves_other_owners_and_reclaims_empty_records() {
        let directory = CheckpointDirectory::default();
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let second = ResidencyOwner::worker(WorkerWithDpRank::new(8, 0));
        directory.publish(first, vec![(11, 0)]);
        assert!(directory.is_empty());
        directory.publish(first, vec![(11, 1)]);
        directory.publish(second, vec![(11, 2)]);
        directory.publish(first, vec![(11, 0)]);
        {
            let entry = directory.entries.get(&11).unwrap();
            let record = entry.value();
            assert!(record.device_workers.is_empty());
            assert!(record.owners[0].is_empty());
            assert_eq!(record.owners[1], FxHashSet::from_iter([second]));
        }
        directory.publish(second, vec![(11, 0)]);
        assert!(directory.is_empty());
        directory.publish(first, vec![(11, 1)]);
        assert_eq!(
            directory.entries.get(&11).unwrap().device_workers.workers,
            FxHashSet::from_iter([WorkerWithDpRank::new(7, 0)])
        );
    }

    #[test]
    fn checkpoint_snapshot_readers_share_the_shard_lock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let table = Arc::new(KvCacheGroupResidency::new());
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let reader = table.checkpoints.entries.get(&hash.0).unwrap();
        let (send, receive) = mpsc::channel();
        let shared = table.clone();
        let thread = std::thread::spawn(move || {
            assert_eq!(
                shared.checkpoint_workers(hash).unwrap(),
                FxHashSet::from_iter([worker])
            );
            let found = shared
                .checkpoints
                .with_owners(hash, StorageTier::Device, |owners| {
                    assert_eq!(owners.len(), 1);
                    assert!(owners.contains(&owner));
                });
            send.send(found).unwrap();
        });
        let result = receive.recv_timeout(Duration::from_secs(5));
        drop(reader);
        thread.join().unwrap();
        assert!(result.expect("snapshot readers of the same shard must coexist"));
    }

    #[test]
    fn held_workers_snapshot_does_not_block_updates_or_change_its_contents() {
        use std::sync::mpsc;
        use std::time::Duration;
        let table = Arc::new(KvCacheGroupResidency::new());
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let hash = ExternalSequenceBlockHash(11);
        table.apply(&checkpoint(first, StorageTier::Device, 0, 11, Some(1)));
        let captured = table.checkpoint_workers(hash).unwrap();
        let (send, receive) = mpsc::channel();
        let shared = table.clone();
        let thread = std::thread::spawn(move || {
            shared.apply(&checkpoint(second, StorageTier::Device, 0, 11, Some(1)));
            send.send(()).unwrap();
        });
        let result = receive.recv_timeout(Duration::from_secs(5));
        thread.join().unwrap();
        result.expect("retaining a worker snapshot must not block its writer");
        assert_eq!(captured, FxHashSet::from_iter([first]));
        assert_eq!(
            table.checkpoint_workers(hash).unwrap(),
            FxHashSet::from_iter([first, second])
        );
    }

    #[test]
    fn owner_and_worker_snapshot_callbacks_run_after_unlocking() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let hash = ExternalSequenceBlockHash(11);
        table.apply(&checkpoint(first, StorageTier::Device, 0, 11, Some(1)));
        assert!(table.with_checkpoint_workers(hash, |workers| {
            assert!(matches!(
                table.checkpoints.entries.try_get_mut(&hash.0),
                dashmap::try_result::TryResult::Present(_)
            ));
            table.apply(&checkpoint(second, StorageTier::Device, 0, 11, Some(1)));
            assert_eq!(workers, &FxHashSet::from_iter([first]));
        }));
        assert!(
            table
                .read_view()
                .with_owners(hash, StorageTier::Device, |owners| {
                    assert!(matches!(
                        table.checkpoints.entries.try_get_mut(&hash.0),
                        dashmap::try_result::TryResult::Present(_)
                    ));
                    assert_eq!(owners.len(), 2);
                    table.clear_worker(first);
                    assert_eq!(owners.len(), 2);
                    assert!(owners.contains(&ResidencyOwner::worker(first)));
                })
        );
        assert_eq!(
            table.checkpoint_workers(hash).unwrap(),
            FxHashSet::from_iter([second])
        );
    }

    #[test]
    fn concurrent_delete_readd_preserves_shared_and_independent_memberships() {
        use std::sync::Barrier;
        let directory = Arc::new(CheckpointDirectory::default());
        let barrier = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            for id in 0..8 {
                let directory = directory.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let owner = ResidencyOwner::worker(WorkerWithDpRank::new(id, 0));
                    let updates = (0..256)
                        .map(|offset| (id * 256 + offset, 1))
                        .chain([(u64::MAX, 1)])
                        .collect::<Vec<_>>();
                    barrier.wait();
                    for _ in 0..3 {
                        directory.publish(owner, updates.clone());
                        directory
                            .publish(owner, updates.iter().map(|&(hash, _)| (hash, 0)).collect());
                    }
                    directory.publish(owner, updates);
                });
            }
        });
        assert_eq!(directory.entries.len(), 2049);
        for hash in 0..2048 {
            let entry = directory.entries.get(&hash).unwrap();
            let record = entry.value();
            let worker = WorkerWithDpRank::new(hash / 256, 0);
            assert_eq!(
                record.owners[0],
                FxHashSet::from_iter([ResidencyOwner::worker(worker)])
            );
            assert_eq!(
                record.device_workers.workers,
                FxHashSet::from_iter([worker])
            );
        }
        let entry = directory.entries.get(&u64::MAX).unwrap();
        let shared = entry.value();
        assert_eq!(shared.owners[0].len(), 8);
        assert_eq!(
            shared.device_workers.workers,
            (0..8).map(|id| WorkerWithDpRank::new(id, 0)).collect()
        );
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
        assert!(table.checkpoints.is_empty());
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
            table.with_checkpoint_workers(ExternalSequenceBlockHash(11), |_| {});
        }
        for writer in writers {
            writer.join().unwrap();
        }
        assert!(!table.is_active());
        assert!(table.checkpoints.is_empty());
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
        let mut view = table.read_view();
        view.track_owner(ResidencyOwner::worker(first));
        assert_eq!(
            view.owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>(),
            vec![ResidencyOwner::worker(first)]
        );
        table.remove_worker_rank(first);
        assert!(table.with_checkpoint_workers(hash, |workers| {
            assert_eq!(workers.len(), 1);
            assert!(workers.contains(&second));
        }));
        assert!(
            view.owners_at(hash, StorageTier::HostPinned)
                .next()
                .is_none()
        );
        table.remove_worker(7);
        assert!(table.checkpoints.is_empty());
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
        assert!(table.checkpoints.is_empty());
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
}
