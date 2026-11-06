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
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;

fn tier_bit(tier: StorageTier) -> u8 {
    match tier {
        StorageTier::Device => 1,
        StorageTier::HostPinned => 2,
        StorageTier::Disk => 4,
        StorageTier::External => 8,
    }
}

#[derive(Debug, Default)]
struct OwnerPresence {
    groups: GroupPresence,
    declared: bool,
    expected_group_count: Option<usize>,
    alignment_blocks: usize,
}

/// Recurrent checkpoints are separate from attention edges and scoped to their owner.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, Arc<RwLock<OwnerPresence>>>,
    cache_owners: FxHashMap<ResidencyOwnerKey, Arc<RwLock<OwnerPresence>>>,
}

#[derive(Debug, Default)]
pub struct KvCacheGroupResidency {
    registry: ArcSwap<OwnerRegistry>,
    registry_writer: Mutex<()>,
    active: AtomicBool,
}

impl KvCacheGroupResidency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    fn owner(&self, owner: ResidencyOwner) -> Option<Arc<RwLock<OwnerPresence>>> {
        self.registry.load().owners.get(&owner).cloned()
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
        let state = Arc::new(RwLock::new(OwnerPresence::default()));
        next.owners.insert(owner, state.clone());
        if matches!(owner, ResidencyOwner::CacheOwner(_)) {
            next.cache_owners.insert(owner.compact_key(), state.clone());
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
                if state.declared && !state.groups.contains_key(&tag.group_idx) {
                    return true;
                }
                if let Some(count) = tag.expected_group_count {
                    if count == 0
                        || state
                            .expected_group_count
                            .is_some_and(|expected| expected != count as usize)
                    {
                        return true;
                    }
                    state.expected_group_count = Some(count as usize);
                }
                if let Some(alignment) = tag.alignment_blocks {
                    let alignment = alignment.max(1) as usize;
                    if state.alignment_blocks != 0 && state.alignment_blocks != alignment {
                        return true;
                    }
                    state.alignment_blocks = alignment;
                }
                let group = state.groups.entry(tag.group_idx).or_default();
                for block in &data.blocks {
                    *group.entry(block.block_hash.0).or_insert(0) |= tier_bit(event.storage_tier);
                }
            }
            KvCacheEventData::Removed(data) => {
                if let Some(state) = self.owner(owner) {
                    let mut state = state.write();
                    if let Some(group) = state.groups.get_mut(&tag.group_idx) {
                        for hash in &data.block_hashes {
                            if let Some(mask) = group.get_mut(&hash.0) {
                                *mask &= !tier_bit(event.storage_tier);
                                if *mask == 0 {
                                    group.remove(&hash.0);
                                }
                            }
                        }
                    }
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
                for group in state.write().groups.values_mut() {
                    group.clear();
                }
            }
        }
    }

    pub fn clear_worker(&self, worker: WorkerWithDpRank) {
        if let Some(state) = self.owner(ResidencyOwner::worker(worker)) {
            for group in state.write().groups.values_mut() {
                group.clear();
            }
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
        next.owners.retain(
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

    /// Evaluate checkpoint membership with one owner read guard per candidate path.
    pub fn reusable_blocks_on_tier(
        &self,
        owner: ResidencyOwner,
        chain: &[ExternalSequenceBlockHash],
        blocks: usize,
        tier: StorageTier,
        floor: usize,
    ) -> Option<usize> {
        let state = self.owner(owner)?;
        Self::evaluate_state(&state.read(), chain, blocks, tier, floor)
    }

    pub fn reusable_cache_owner_blocks(
        &self,
        owner: ResidencyOwnerKey,
        chain: &[ExternalSequenceBlockHash],
        blocks: usize,
        tier: StorageTier,
        floor: usize,
    ) -> Option<usize> {
        let state = self.registry.load().cache_owners.get(&owner).cloned()?;
        Self::evaluate_state(&state.read(), chain, blocks, tier, floor)
    }

    fn evaluate_state(
        state: &OwnerPresence,
        chain: &[ExternalSequenceBlockHash],
        blocks: usize,
        tier: StorageTier,
        floor: usize,
    ) -> Option<usize> {
        if state.groups.is_empty() {
            return Some(0);
        }
        if state.expected_group_count != Some(state.groups.len()) {
            return Some(0);
        }
        let limit = blocks.min(chain.len());
        Some(
            (floor.saturating_add(1)..=limit)
                .rev()
                .find(|&p| {
                    p % state.alignment_blocks.max(1) == 0
                        && state.groups.values().all(|group| {
                            group
                                .get(&chain[p - 1].0)
                                .is_some_and(|mask| mask & tier_bit(tier) != 0)
                        })
                })
                .unwrap_or(floor.min(limit)),
        )
    }

    pub fn is_hybrid(&self, worker: WorkerWithDpRank) -> bool {
        self.owner(ResidencyOwner::worker(worker)).is_some()
    }

    pub fn reusable_blocks(
        &self,
        worker: WorkerWithDpRank,
        chain: &[ExternalSequenceBlockHash],
        blocks: usize,
    ) -> Option<usize> {
        self.reusable_blocks_on_tier(
            ResidencyOwner::worker(worker),
            chain,
            blocks,
            StorageTier::Device,
            0,
        )
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

    #[test]
    fn malformed_zero_group_count_never_enables_dense_fallback() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(0)));
        assert_eq!(
            table.reusable_blocks(worker, &[ExternalSequenceBlockHash(11)], 1),
            Some(0)
        );
    }

    #[test]
    fn checkpoint_cut_respects_engine_group_alignment() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.declare_layout(ResidencyOwner::worker(worker), &[0], 3);
        for hash in [2, 3] {
            let mut event = checkpoint(worker, StorageTier::Device, 0, hash, Some(1));
            event.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(3);
            table.apply(&event);
        }
        let chain: Vec<_> = (1..=4).map(ExternalSequenceBlockHash).collect();
        assert_eq!(table.reusable_blocks(worker, &chain, 4), Some(3));
        assert_eq!(table.reusable_blocks(worker, &chain, 2), Some(0));
    }

    #[test]
    fn incomplete_layout_never_grants_credit_and_dump_preserves_empty_groups() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_groups(owner, &[0, 1, 2]);
        let chain = [ExternalSequenceBlockHash(11)];
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(3)));
        }
        assert_eq!(table.reusable_blocks(worker, &chain, 1), Some(0));
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        assert_eq!(restored.groups_for(worker).len(), 3);
        assert_eq!(restored.reusable_blocks(worker, &chain, 1), Some(0));
        restored.apply(&checkpoint(worker, StorageTier::Device, 2, 11, Some(3)));
        assert_eq!(restored.reusable_blocks(worker, &chain, 1), Some(1));
    }

    #[test]
    fn tier_membership_and_removal_are_independent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let chain = [ExternalSequenceBlockHash(11)];
        let mut event = checkpoint(worker, StorageTier::HostPinned, 0, 11, Some(1));
        table.apply(&event);
        assert_eq!(table.reusable_blocks(worker, &chain, 1), Some(0));
        assert_eq!(
            table.reusable_blocks_on_tier(owner, &chain, 1, StorageTier::HostPinned, 0),
            Some(1)
        );
        event.storage_tier = StorageTier::Device;
        table.apply(&event);
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: chain.to_vec(),
        });
        table.apply(&event);
        assert_eq!(table.reusable_blocks(worker, &chain, 1), Some(0));
        assert_eq!(
            table.reusable_blocks_on_tier(owner, &chain, 1, StorageTier::HostPinned, 0),
            Some(1)
        );
    }

    #[test]
    fn missing_completeness_metadata_fails_closed() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, None));
        assert_eq!(
            table.reusable_blocks(worker, &[ExternalSequenceBlockHash(11)], 1),
            Some(0)
        );
        table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        assert_eq!(
            table.reusable_blocks(worker, &[ExternalSequenceBlockHash(11)], 1),
            Some(1)
        );
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
        assert_eq!(
            table.reusable_blocks(worker, &[ExternalSequenceBlockHash(11)], 1),
            Some(0)
        );
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
}
