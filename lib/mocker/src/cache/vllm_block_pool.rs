// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical-capacity model for vLLM's GPU block pool.
//!
//! A cached hash may have several physical copies. Copy identity is internal;
//! the pool models occupancy, reference/pin state, and LRU eviction without
//! reproducing vLLM's numeric block IDs or null block.

use dynamo_tokens::SequenceHash;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};
use std::{
    collections::{VecDeque, hash_map::Entry},
    sync::{Arc, OnceLock},
};

new_key_type! {
    pub(crate) struct BlockCopyId;
}

#[derive(Debug)]
enum CopyState {
    Private,
    /// A cached copy is linked into the inactive LRU if and only if both
    /// `refs` and `pins` are zero. Any future cached sub-state must preserve or
    /// explicitly revise that membership invariant.
    Cached {
        hash: SequenceHash,
        refs: usize,
        pins: usize,
        inactive_prev: Option<BlockCopyId>,
        inactive_next: Option<BlockCopyId>,
    },
}

#[derive(Debug)]
struct BlockCopy {
    state: CopyState,
}

struct HashCopies {
    primary: BlockCopyId,
    // Keep the common one-copy value to two machine words; duplicate hashes
    // pay the extra allocation only on the uncommon overflow path.
    #[allow(clippy::box_collection)]
    duplicates: Option<Box<VecDeque<BlockCopyId>>>,
}

enum CopyRemoval {
    Last,
    Remaining,
    Missing,
}

impl HashCopies {
    fn new(primary: BlockCopyId) -> Self {
        Self {
            primary,
            duplicates: None,
        }
    }

    fn push(&mut self, id: BlockCopyId) {
        self.duplicates
            .get_or_insert_with(|| Box::new(VecDeque::new()))
            .push_back(id);
    }

    fn remove(&mut self, id: BlockCopyId) -> CopyRemoval {
        if self.primary == id {
            let Some(duplicates) = self.duplicates.as_mut() else {
                return CopyRemoval::Last;
            };
            self.primary = duplicates
                .pop_front()
                .expect("duplicate-copy queue must not be empty");
            if duplicates.is_empty() {
                self.duplicates = None;
            }
            return CopyRemoval::Remaining;
        }

        let Some(duplicates) = self.duplicates.as_mut() else {
            return CopyRemoval::Missing;
        };
        let Some(position) = duplicates.iter().position(|candidate| *candidate == id) else {
            return CopyRemoval::Missing;
        };
        duplicates.remove(position);
        if duplicates.is_empty() {
            self.duplicates = None;
        }
        CopyRemoval::Remaining
    }

    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = BlockCopyId> + '_ {
        std::iter::once(self.primary).chain(
            self.duplicates
                .iter()
                .flat_map(|duplicates| duplicates.iter().copied()),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixHit {
    pub(crate) is_active: bool,
}

/// Capacity and cached-prefix pins held before a manager commits ownership.
pub(crate) struct BlockReservation {
    /// Cached prefix copies in request order, from root/head to suffix/leaf.
    prefix: Vec<(SequenceHash, BlockCopyId)>,
    fresh: usize,
}

impl BlockReservation {
    pub(crate) fn len(&self) -> usize {
        self.prefix.len() + self.fresh
    }

    pub(crate) fn fresh_len(&self) -> usize {
        self.fresh
    }
}

pub(crate) struct ReserveOutcome {
    pub(crate) reservation: BlockReservation,
    /// Hashes whose final cache-visible physical copy was evicted.
    pub(crate) removed: Vec<SequenceHash>,
}

pub(crate) struct VllmBlockPool {
    capacity: usize,
    copies: SlotMap<BlockCopyId, BlockCopy>,
    by_hash: FxHashMap<SequenceHash, HashCopies>,
    /// Immutable view of the exact `by_hash` key set. Prefix snapshots may
    /// retain an older `Arc` while an eager scheduler pass mutates the pool.
    resident_hashes: OnceLock<Arc<FxHashSet<SequenceHash>>>,
    /// Intrusive ordinary LRU: head is evicted first, tail was released last.
    inactive_head: Option<BlockCopyId>,
    inactive_tail: Option<BlockCopyId>,
    inactive_len: usize,
    reserved: usize,
}

impl VllmBlockPool {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            copies: SlotMap::with_key(),
            by_hash: FxHashMap::default(),
            resident_hashes: OnceLock::new(),
            inactive_head: None,
            inactive_tail: None,
            inactive_len: 0,
            reserved: 0,
        }
    }

    pub(crate) fn prefix_hit(&self, hash: SequenceHash) -> Option<PrefixHit> {
        let id = self.first_copy(hash)?;
        let copy = &self.copies[id];
        let CopyState::Cached { refs, pins, .. } = &copy.state else {
            unreachable!("hash index points to a private copy")
        };
        Some(PrefixHit {
            is_active: *refs > 0 || *pins > 0,
        })
    }

    pub(crate) fn resident_hashes(&self) -> Arc<FxHashSet<SequenceHash>> {
        Arc::clone(
            self.resident_hashes
                .get_or_init(|| Arc::new(self.by_hash.keys().copied().collect())),
        )
    }

    fn invalidate_resident_hashes(&mut self) {
        self.resident_hashes.take();
    }

    /// Atomically pins `prefix` and reserves `fresh` additional copies.
    ///
    /// The caller obtains `prefix` from a preceding synchronous lookup. A
    /// missing hash is therefore an invariant violation rather than capacity
    /// exhaustion.
    pub(crate) fn reserve(
        &mut self,
        prefix: &[SequenceHash],
        fresh: usize,
    ) -> Option<ReserveOutcome> {
        if prefix.is_empty() {
            return self.reserve_fresh(fresh);
        }

        self.reserve_exact_prefix(prefix.iter().copied(), prefix.len() + fresh)
    }

    /// Resolve and pin the longest resident prefix from `candidates`, then
    /// reserve the remaining entries as fresh capacity in one traversal.
    pub(crate) fn reserve_resident_prefix(
        &mut self,
        candidates: impl IntoIterator<Item = SequenceHash>,
        total: usize,
    ) -> Option<ReserveOutcome> {
        let mut candidates = candidates.into_iter();
        let Some(first_hash) = candidates.next() else {
            return self.reserve_fresh(total);
        };
        assert!(total > 0, "prefix candidates exceed layout");
        let Some(first_id) = self.first_copy(first_hash) else {
            return self.reserve_fresh(total);
        };

        let mut hits = vec![(first_hash, first_id)];
        for hash in candidates {
            assert!(hits.len() < total, "prefix candidates exceed layout");
            let Some(id) = self.first_copy(hash) else {
                break;
            };
            hits.push((hash, id));
        }
        let fresh = total - hits.len();
        self.reserve_hits(hits, fresh)
    }

    /// Pin an already-authorized prefix and reserve the remaining entries.
    ///
    /// Unlike [`Self::reserve_resident_prefix`], every candidate must still be
    /// resident. A missing hash means the caller's synchronous prefix
    /// authorization changed before allocation committed.
    pub(crate) fn reserve_exact_prefix<I>(
        &mut self,
        candidates: I,
        total: usize,
    ) -> Option<ReserveOutcome>
    where
        I: IntoIterator<Item = SequenceHash>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut candidates = candidates.into_iter();
        let candidate_count = candidates.len();
        let Some(first_hash) = candidates.next() else {
            return self.reserve_fresh(total);
        };
        assert!(candidate_count <= total, "prefix candidates exceed layout");
        let Some(first_id) = self.first_copy(first_hash) else {
            panic!("authorized prefix hash {first_hash} is no longer resident")
        };
        let mut hits = Vec::with_capacity(candidate_count);
        hits.push((first_hash, first_id));

        for hash in candidates {
            assert!(hits.len() < total, "prefix candidates exceed layout");
            let Some(id) = self.first_copy(hash) else {
                panic!("authorized prefix hash {hash} is no longer resident")
            };
            hits.push((hash, id));
        }
        let fresh = total - hits.len();
        self.reserve_hits(hits, fresh)
    }

    fn reserve_hits(
        &mut self,
        hits: Vec<(SequenceHash, BlockCopyId)>,
        fresh: usize,
    ) -> Option<ReserveOutcome> {
        let free = self.free_capacity();
        let needed_evictions = fresh.saturating_sub(free);
        if needed_evictions > 0 {
            let inactive_hits = hits
                .iter()
                .filter_map(|(_, id)| self.is_inactive(*id).then_some(*id))
                .collect::<FxHashSet<_>>()
                .len();
            let evictable_after_pins = self.inactive_len.saturating_sub(inactive_hits);
            if needed_evictions > evictable_after_pins {
                return None;
            }
        }

        for (_, id) in &hits {
            self.pin(*id);
        }

        let mut removed = Vec::with_capacity(needed_evictions);
        for _ in 0..needed_evictions {
            if let Some(hash) = self.evict_one() {
                removed.push(hash);
            }
        }
        self.reserved += fresh;

        Some(ReserveOutcome {
            reservation: BlockReservation {
                prefix: hits,
                fresh,
            },
            removed,
        })
    }

    fn reserve_fresh(&mut self, fresh: usize) -> Option<ReserveOutcome> {
        let free = self.free_capacity();
        let needed_evictions = fresh.saturating_sub(free);
        if needed_evictions > self.inactive_len {
            return None;
        }

        let mut removed = Vec::with_capacity(needed_evictions);
        for _ in 0..needed_evictions {
            if let Some(hash) = self.evict_one() {
                removed.push(hash);
            }
        }
        self.reserved += fresh;

        Some(ReserveOutcome {
            reservation: BlockReservation {
                prefix: Vec::new(),
                fresh,
            },
            removed,
        })
    }

    /// Convert all cached-prefix pins into request references.
    pub(crate) fn activate_prefix(
        &mut self,
        reservation: &mut BlockReservation,
    ) -> std::vec::IntoIter<(SequenceHash, BlockCopyId)> {
        let prefix = std::mem::take(&mut reservation.prefix);
        for &(hash, id) in &prefix {
            self.activate_pin(id, hash);
        }
        prefix.into_iter()
    }

    pub(crate) fn allocate_private(&mut self, reservation: &mut BlockReservation) -> BlockCopyId {
        assert!(reservation.fresh > 0, "reservation has no fresh capacity");
        assert!(self.reserved > 0, "pool reserved-capacity underflow");
        reservation.fresh -= 1;
        self.reserved -= 1;

        self.copies.insert(BlockCopy {
            state: CopyState::Private,
        })
    }

    /// Allocate a transferred/computed full block directly into the cache.
    /// Returns whether the hash became router-visible (`0 -> 1`).
    pub(crate) fn allocate_cached(
        &mut self,
        reservation: &mut BlockReservation,
        hash: SequenceHash,
    ) -> (BlockCopyId, bool) {
        let id = self.allocate_private(reservation);
        let became_visible = self.cache_private(id, hash);
        (id, became_visible)
    }

    /// Make a request-private computed full block available for prefix reuse.
    /// Returns whether this is the first resident physical copy of `hash`.
    pub(crate) fn cache_private(&mut self, id: BlockCopyId, hash: SequenceHash) -> bool {
        let Some(copy) = self.copies.get_mut(id) else {
            panic!("attempted to cache an unknown block copy")
        };
        assert!(
            matches!(copy.state, CopyState::Private),
            "only a private copy can enter the prefix cache"
        );
        copy.state = CopyState::Cached {
            hash,
            refs: 1,
            pins: 0,
            inactive_prev: None,
            inactive_next: None,
        };
        let became_visible = match self.by_hash.entry(hash) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().push(id);
                false
            }
            Entry::Vacant(entry) => {
                entry.insert(HashCopies::new(id));
                true
            }
        };
        if became_visible {
            self.invalidate_resident_hashes();
        }
        became_visible
    }

    /// Release one request-owned reference. Private copies return capacity
    /// immediately; cached copies become inactive LRU candidates at refcount 0.
    pub(crate) fn release(&mut self, id: BlockCopyId) {
        let Some(copy) = self.copies.get(id) else {
            panic!("attempted to release an unknown block copy")
        };
        if matches!(copy.state, CopyState::Private) {
            self.copies.remove(id);
            return;
        }

        let should_deactivate = {
            let CopyState::Cached { refs, pins, .. } = &mut self.copies[id].state else {
                unreachable!()
            };
            assert!(*refs > 0, "cached-copy reference underflow");
            *refs -= 1;
            *refs == 0 && *pins == 0
        };
        if should_deactivate {
            self.insert_inactive(id);
        }
    }

    /// Release all unconsumed capacity and prefix pins.
    ///
    /// Prefix reservations are stored head-to-tail, while the pool expects
    /// callers to release them in eviction-priority order. Unpinning in reverse
    /// makes suffix/leaf blocks older LRU candidates than their parents.
    pub(crate) fn cancel(&mut self, reservation: BlockReservation) {
        for (hash, id) in reservation.prefix.into_iter().rev() {
            self.unpin(id, hash);
        }
        assert!(
            self.reserved >= reservation.fresh,
            "pool reserved-capacity underflow"
        );
        self.reserved -= reservation.fresh;
    }

    pub(crate) fn num_active(&self) -> usize {
        self.copies.len() - self.inactive_len + self.reserved
    }

    pub(crate) fn num_active_refs(&self) -> usize {
        self.copies
            .values()
            .map(|copy| match &copy.state {
                CopyState::Private => 1,
                CopyState::Cached { refs, .. } => *refs,
            })
            .sum()
    }

    pub(crate) fn num_inactive(&self) -> usize {
        self.inactive_len
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    fn free_capacity(&self) -> usize {
        self.capacity
            .checked_sub(self.copies.len() + self.reserved)
            .unwrap_or_else(|| panic!("block-pool occupancy exceeds capacity"))
    }

    fn first_copy(&self, hash: SequenceHash) -> Option<BlockCopyId> {
        self.by_hash.get(&hash).map(|copies| copies.primary)
    }

    fn is_inactive(&self, id: BlockCopyId) -> bool {
        let CopyState::Cached { refs, pins, .. } = &self.copies[id].state else {
            return false;
        };
        *refs == 0 && *pins == 0
    }

    fn pin(&mut self, id: BlockCopyId) {
        // Must unlink before bumping pins: list membership is derived from
        // refs and pins.
        if self.is_inactive(id) {
            self.unlink_inactive(id);
        }
        let CopyState::Cached { pins, .. } = &mut self.copies[id].state else {
            panic!("prefix hash points to a private copy")
        };
        *pins = pins
            .checked_add(1)
            .unwrap_or_else(|| panic!("pin count overflow"));
    }

    fn activate_pin(&mut self, id: BlockCopyId, expected_hash: SequenceHash) {
        let CopyState::Cached {
            hash, refs, pins, ..
        } = &mut self.copies[id].state
        else {
            panic!("prefix reservation points to a private copy")
        };
        assert_eq!(*hash, expected_hash, "reserved prefix hash changed");
        assert!(*pins > 0, "prefix pin underflow");
        *pins -= 1;
        *refs = refs
            .checked_add(1)
            .unwrap_or_else(|| panic!("reference count overflow"));
    }

    fn unpin(&mut self, id: BlockCopyId, expected_hash: SequenceHash) {
        let should_deactivate = {
            let CopyState::Cached {
                hash, refs, pins, ..
            } = &mut self.copies[id].state
            else {
                panic!("prefix reservation points to a private copy")
            };
            assert_eq!(*hash, expected_hash, "reserved prefix hash changed");
            assert!(*pins > 0, "prefix pin underflow");
            *pins -= 1;
            *pins == 0 && *refs == 0
        };
        if should_deactivate {
            self.insert_inactive(id);
        }
    }

    fn insert_inactive(&mut self, id: BlockCopyId) {
        debug_assert!(
            self.is_inactive(id),
            "only an unreferenced, unpinned cached copy can enter the inactive LRU"
        );
        // A singleton has no links, so head membership is its only
        // double-insertion signal.
        debug_assert_ne!(
            self.inactive_head,
            Some(id),
            "copy is already in the inactive LRU"
        );
        let previous_tail = self.inactive_tail;
        {
            let (prev, next) = self.inactive_links_mut(id);
            debug_assert!(
                prev.is_none() && next.is_none(),
                "copy entering the inactive LRU still has list links"
            );
            *prev = previous_tail;
        }
        if let Some(tail) = previous_tail {
            let (_, next) = self.inactive_links_mut(tail);
            let old_next = next.replace(id);
            debug_assert!(
                old_next.is_none(),
                "inactive LRU tail already has a successor"
            );
        } else {
            let old_head = self.inactive_head.replace(id);
            debug_assert!(old_head.is_none(), "empty inactive LRU still has a head");
        }
        self.inactive_tail = Some(id);
        self.inactive_len = self
            .inactive_len
            .checked_add(1)
            .unwrap_or_else(|| panic!("inactive block count overflow"));
    }

    fn inactive_links_mut(
        &mut self,
        id: BlockCopyId,
    ) -> (&mut Option<BlockCopyId>, &mut Option<BlockCopyId>) {
        let CopyState::Cached {
            inactive_prev,
            inactive_next,
            ..
        } = &mut self.copies[id].state
        else {
            panic!("inactive LRU link target is not a cached copy")
        };
        (inactive_prev, inactive_next)
    }

    fn unlink_inactive(&mut self, id: BlockCopyId) {
        debug_assert!(
            self.is_inactive(id),
            "only an unreferenced, unpinned cached copy can leave the inactive LRU"
        );
        let (previous, next) = {
            let (previous, next) = self.inactive_links_mut(id);
            (previous.take(), next.take())
        };

        if let Some(previous) = previous {
            let (_, previous_next) = self.inactive_links_mut(previous);
            let old_next = std::mem::replace(previous_next, next);
            debug_assert_eq!(
                old_next,
                Some(id),
                "inactive LRU predecessor does not point to the removed copy"
            );
        } else {
            let old_head = std::mem::replace(&mut self.inactive_head, next);
            debug_assert_eq!(
                old_head,
                Some(id),
                "inactive LRU head does not match the removed copy"
            );
        }

        if let Some(next) = next {
            let (next_previous, _) = self.inactive_links_mut(next);
            let old_previous = std::mem::replace(next_previous, previous);
            debug_assert_eq!(
                old_previous,
                Some(id),
                "inactive LRU successor does not point to the removed copy"
            );
        } else {
            let old_tail = std::mem::replace(&mut self.inactive_tail, previous);
            debug_assert_eq!(
                old_tail,
                Some(id),
                "inactive LRU tail does not match the removed copy"
            );
        }

        self.inactive_len = self
            .inactive_len
            .checked_sub(1)
            .unwrap_or_else(|| panic!("inactive block count underflow"));
    }

    #[cfg(test)]
    fn assert_lru_consistent(&self) {
        self.assert_hash_index_consistent();

        let mut linked = FxHashSet::default();
        let mut previous = None;
        let mut cursor = self.inactive_head;

        while let Some(id) = cursor {
            assert!(linked.insert(id), "inactive LRU contains a cycle");
            let Some(copy) = self.copies.get(id) else {
                panic!("inactive LRU points to a missing copy")
            };
            let CopyState::Cached {
                refs,
                pins,
                inactive_prev,
                inactive_next,
                ..
            } = &copy.state
            else {
                panic!("inactive LRU contains a private copy")
            };
            assert_eq!(
                (*refs, *pins),
                (0, 0),
                "active copy is linked into the inactive LRU"
            );
            assert_eq!(
                *inactive_prev, previous,
                "inactive LRU contains a broken back-pointer"
            );
            previous = Some(id);
            cursor = *inactive_next;
        }

        assert_eq!(
            linked.len(),
            self.inactive_len,
            "inactive LRU length does not match its reachable copies"
        );
        assert_eq!(
            self.inactive_tail, previous,
            "inactive LRU tail does not match its final reachable copy"
        );
        assert_eq!(
            self.inactive_head.is_none(),
            self.inactive_tail.is_none(),
            "inactive LRU head and tail emptiness disagree"
        );

        for (id, copy) in self.copies.iter() {
            let CopyState::Cached {
                refs,
                pins,
                inactive_prev,
                inactive_next,
                ..
            } = &copy.state
            else {
                assert!(!linked.contains(&id), "private copy is in the inactive LRU");
                continue;
            };
            let should_be_linked = *refs == 0 && *pins == 0;
            assert_eq!(
                linked.contains(&id),
                should_be_linked,
                "cached copy membership disagrees with its refs and pins"
            );
            if !should_be_linked {
                assert!(
                    inactive_prev.is_none() && inactive_next.is_none(),
                    "active cached copy retains inactive LRU links"
                );
            }
        }
    }

    #[cfg(test)]
    fn assert_hash_index_consistent(&self) {
        if let Some(resident_hashes) = self.resident_hashes.get() {
            assert_eq!(
                resident_hashes.len(),
                self.by_hash.len(),
                "resident-hash snapshot has the wrong cardinality"
            );
            assert!(
                self.by_hash
                    .keys()
                    .all(|hash| resident_hashes.contains(hash)),
                "resident-hash snapshot differs from the hash index"
            );
        }

        let mut indexed = FxHashSet::default();
        for (&expected_hash, copies) in &self.by_hash {
            for id in copies.iter() {
                assert!(indexed.insert(id), "copy is indexed by multiple hashes");
                let Some(copy) = self.copies.get(id) else {
                    panic!("hash index points to a missing copy")
                };
                let CopyState::Cached { hash, .. } = &copy.state else {
                    panic!("hash index points to a private copy")
                };
                assert_eq!(*hash, expected_hash, "copy is indexed under the wrong hash");
            }
        }

        for (id, copy) in self.copies.iter() {
            match &copy.state {
                CopyState::Private => {
                    assert!(!indexed.contains(&id), "private copy is hash-indexed");
                }
                CopyState::Cached { .. } => {
                    assert!(indexed.contains(&id), "cached copy is not hash-indexed");
                }
            }
        }
    }

    /// Evict one physical copy. A hash is returned only on its final copy.
    fn evict_one(&mut self) -> Option<SequenceHash> {
        let Some(id) = self.inactive_head else {
            panic!("prechecked inactive capacity disappeared")
        };
        let CopyState::Cached { inactive_prev, .. } = &self.copies[id].state else {
            panic!("inactive LRU points to a private copy")
        };
        assert!(
            inactive_prev.is_none(),
            "inactive LRU head has a predecessor"
        );
        self.unlink_inactive(id);
        let Some(copy) = self.copies.remove(id) else {
            panic!("inactive LRU points to a missing copy")
        };
        let CopyState::Cached {
            hash, refs, pins, ..
        } = copy.state
        else {
            panic!("inactive LRU points to a private copy")
        };
        assert_eq!(refs, 0, "evicted cached copy still has references");
        assert_eq!(pins, 0, "evicted cached copy is still pinned");

        let remove_hash = {
            let Some(copies) = self.by_hash.get_mut(&hash) else {
                panic!("evicted cached hash is missing from its index")
            };
            match copies.remove(id) {
                CopyRemoval::Last => true,
                CopyRemoval::Remaining => false,
                CopyRemoval::Missing => {
                    panic!("evicted copy is missing from its hash index")
                }
            }
        };
        if remove_hash {
            self.by_hash.remove(&hash);
            self.invalidate_resident_hashes();
            Some(hash)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reserve(pool: &mut VllmBlockPool, prefix: &[u64], fresh: usize) -> ReserveOutcome {
        pool.reserve(prefix, fresh)
            .unwrap_or_else(|| panic!("unexpected capacity exhaustion"))
    }

    #[test]
    #[should_panic(expected = "authorized prefix hash 9 is no longer resident")]
    fn exact_prefix_reports_missing_hash() {
        let mut pool = VllmBlockPool::new(1);
        let _ = pool.reserve_exact_prefix([9], 1);
    }

    #[test]
    fn empty_exact_prefix_reserves_fresh_without_prefix_storage() {
        let mut pool = VllmBlockPool::new(3);
        let outcome = pool
            .reserve_exact_prefix(std::iter::empty(), 3)
            .expect("fresh capacity should fit");

        assert_eq!(outcome.reservation.prefix.capacity(), 0);
        assert_eq!(outcome.reservation.fresh_len(), 3);
        pool.cancel(outcome.reservation);
    }

    #[test]
    fn cold_resident_prefix_reserves_fresh_without_prefix_storage() {
        let mut pool = VllmBlockPool::new(3);
        let outcome = pool
            .reserve_resident_prefix([7, 8, 9], 3)
            .expect("fresh capacity should fit");

        assert_eq!(outcome.reservation.prefix.capacity(), 0);
        assert_eq!(outcome.reservation.fresh_len(), 3);
        pool.cancel(outcome.reservation);
    }

    #[test]
    fn resident_hash_snapshot_changes_only_with_hash_membership() {
        let mut pool = VllmBlockPool::new(2);
        let empty = pool.resident_hashes();
        assert!(Arc::ptr_eq(&empty, &pool.resident_hashes()));

        let mut first = reserve(&mut pool, &[], 1).reservation;
        let first_id = pool.allocate_private(&mut first);
        assert!(pool.cache_private(first_id, 7));
        let visible = pool.resident_hashes();
        assert!(!Arc::ptr_eq(&empty, &visible));
        assert!(empty.is_empty(), "older snapshots must remain frozen");
        assert_eq!(visible.len(), 1);
        assert!(visible.contains(&7));
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));

        let mut duplicate = reserve(&mut pool, &[], 1).reservation;
        let duplicate_id = pool.allocate_private(&mut duplicate);
        assert!(!pool.cache_private(duplicate_id, 7));
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));

        let pinned = reserve(&mut pool, &[7], 0);
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));
        pool.cancel(pinned.reservation);
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));

        pool.release(duplicate_id);
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));
        assert_eq!(pool.evict_one(), None, "one duplicate must remain visible");
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));

        pool.release(first_id);
        assert!(Arc::ptr_eq(&visible, &pool.resident_hashes()));
        assert_eq!(pool.evict_one(), Some(7));
        let removed = pool.resident_hashes();
        assert!(!Arc::ptr_eq(&visible, &removed));
        assert!(removed.is_empty());
        assert!(visible.contains(&7), "older snapshots must remain frozen");
        pool.assert_lru_consistent();
        pool.assert_hash_index_consistent();
    }

    #[test]
    fn duplicate_hashes_consume_distinct_capacity_but_share_visibility() {
        let mut pool = VllmBlockPool::new(2);
        let mut first = reserve(&mut pool, &[], 1).reservation;
        let first_id = pool.allocate_private(&mut first);
        assert!(pool.cache_private(first_id, 7));

        let mut second = reserve(&mut pool, &[], 1).reservation;
        let second_id = pool.allocate_private(&mut second);
        assert!(!pool.cache_private(second_id, 7));
        assert_eq!(pool.num_active(), 2);

        pool.release(first_id);
        pool.release(second_id);
        assert_eq!(pool.num_inactive(), 2);
        pool.assert_lru_consistent();
    }

    #[test]
    fn prefix_pin_is_excluded_from_atomic_fresh_capacity() {
        let mut pool = VllmBlockPool::new(1);
        let mut seed = reserve(&mut pool, &[], 1).reservation;
        let id = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(id, 9));
        pool.release(id);

        assert!(pool.reserve(&[9], 1).is_none());
        assert_eq!(pool.num_active(), 0);
        assert_eq!(pool.num_inactive(), 1);
        pool.assert_lru_consistent();
    }

    #[test]
    fn resident_prefix_stops_at_first_miss_and_reserves_fresh_suffix() {
        let mut pool = VllmBlockPool::new(2);
        let mut seed = reserve(&mut pool, &[], 1).reservation;
        let id = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(id, 7));
        pool.release(id);

        let outcome = pool
            .reserve_resident_prefix([7, 9], 2)
            .expect("one resident prefix plus one fresh block should fit");
        assert!(outcome.removed.is_empty());
        assert_eq!(outcome.reservation.len(), 2);
        assert_eq!(outcome.reservation.fresh_len(), 1);
        assert_eq!(outcome.reservation.prefix.capacity(), 1);
        pool.cancel(outcome.reservation);
        assert_eq!(pool.num_inactive(), 1);
        pool.assert_lru_consistent();
    }

    #[test]
    fn removal_is_reported_only_for_the_last_physical_copy() {
        let mut pool = VllmBlockPool::new(4);
        let mut first = reserve(&mut pool, &[], 1).reservation;
        let first_id = pool.allocate_private(&mut first);
        assert!(pool.cache_private(first_id, 3));

        let mut duplicate_ids = Vec::new();
        for _ in 0..3 {
            let mut duplicate = reserve(&mut pool, &[], 1).reservation;
            let duplicate_id = pool.allocate_private(&mut duplicate);
            assert!(!pool.cache_private(duplicate_id, 3));
            duplicate_ids.push(duplicate_id);
        }
        pool.release(first_id);
        for &duplicate_id in &duplicate_ids {
            pool.release(duplicate_id);
        }

        let first_eviction = reserve(&mut pool, &[], 1);
        assert!(first_eviction.removed.is_empty());
        pool.cancel(first_eviction.reservation);

        let promoted = pool
            .reserve_exact_prefix([3], 1)
            .expect("promoted duplicate should remain reservable");
        assert_eq!(promoted.reservation.prefix, vec![(3, duplicate_ids[0])]);
        pool.cancel(promoted.reservation);

        for fresh in [2, 3] {
            let duplicate_eviction = reserve(&mut pool, &[], fresh);
            assert!(duplicate_eviction.removed.is_empty());
            pool.cancel(duplicate_eviction.reservation);
        }

        let final_eviction = reserve(&mut pool, &[], 4);
        assert_eq!(final_eviction.removed, vec![3]);
        pool.cancel(final_eviction.reservation);
        pool.assert_lru_consistent();
    }

    #[test]
    fn evicting_middle_duplicate_preserves_primary_lookup() {
        let mut pool = VllmBlockPool::new(4);
        let mut first = reserve(&mut pool, &[], 1).reservation;
        let first_id = pool.allocate_private(&mut first);
        assert!(pool.cache_private(first_id, 3));

        let mut duplicate_ids = Vec::new();
        for _ in 0..3 {
            let mut duplicate = reserve(&mut pool, &[], 1).reservation;
            let duplicate_id = pool.allocate_private(&mut duplicate);
            assert!(!pool.cache_private(duplicate_id, 3));
            duplicate_ids.push(duplicate_id);
        }
        let middle_id = duplicate_ids[1];
        pool.release(middle_id);

        let pressure = reserve(&mut pool, &[], 1);
        assert!(pressure.removed.is_empty());
        pool.cancel(pressure.reservation);
        assert!(pool.copies.get(middle_id).is_none());

        let primary = pool
            .reserve_exact_prefix([3], 1)
            .expect("primary copy should remain reservable");
        assert_eq!(primary.reservation.prefix, vec![(3, first_id)]);
        pool.cancel(primary.reservation);
        pool.release(first_id);
        pool.release(duplicate_ids[0]);
        pool.release(duplicate_ids[2]);
        pool.assert_lru_consistent();
    }

    #[test]
    fn canceled_prefix_evicts_leaf_before_parent_under_pressure() {
        let mut pool = VllmBlockPool::new(2);
        let mut seed = reserve(&mut pool, &[], 2).reservation;
        let parent = pool.allocate_private(&mut seed);
        let leaf = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(parent, 7));
        assert!(pool.cache_private(leaf, 8));

        // Match the normal request-release contract: the leaf enters the LRU
        // before its parent.
        pool.release(leaf);
        pool.release(parent);

        let canceled = reserve(&mut pool, &[7, 8], 0);
        assert!(canceled.removed.is_empty());
        pool.cancel(canceled.reservation);

        let pressure = reserve(&mut pool, &[], 1);
        assert_eq!(pressure.removed, vec![8]);
        assert!(pool.prefix_hit(7).is_some());
        assert!(pool.prefix_hit(8).is_none());
        pool.cancel(pressure.reservation);
        pool.assert_lru_consistent();
    }

    #[test]
    fn pinning_middle_inactive_copy_preserves_lru_order() {
        let mut pool = VllmBlockPool::new(3);
        let mut seed = reserve(&mut pool, &[], 3).reservation;
        let first = pool.allocate_private(&mut seed);
        let middle = pool.allocate_private(&mut seed);
        let last = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(first, 1));
        assert!(pool.cache_private(middle, 2));
        assert!(pool.cache_private(last, 3));
        pool.release(first);
        pool.release(middle);
        pool.release(last);
        pool.assert_lru_consistent();

        let pinned = reserve(&mut pool, &[2], 1);
        assert_eq!(pinned.removed, vec![1]);
        pool.assert_lru_consistent();
        pool.cancel(pinned.reservation);
        pool.assert_lru_consistent();

        let pressure = reserve(&mut pool, &[], 2);
        assert_eq!(pressure.removed, vec![3]);
        pool.assert_lru_consistent();
        pool.cancel(pressure.reservation);
        pool.assert_lru_consistent();
    }

    #[test]
    fn activated_prefix_reenters_inactive_lru_on_release() {
        let mut pool = VllmBlockPool::new(1);
        let mut seed = reserve(&mut pool, &[], 1).reservation;
        let id = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(id, 7));
        pool.release(id);

        let mut activation = reserve(&mut pool, &[7], 0).reservation;
        assert_eq!(
            pool.activate_prefix(&mut activation).collect::<Vec<_>>(),
            vec![(7, id)]
        );
        pool.cancel(activation);
        assert_eq!(pool.num_inactive(), 0);
        pool.assert_lru_consistent();

        pool.release(id);
        assert_eq!(pool.num_inactive(), 1);
        pool.assert_lru_consistent();

        let pressure = reserve(&mut pool, &[], 1);
        assert_eq!(pressure.removed, vec![7]);
        pool.cancel(pressure.reservation);
        pool.assert_lru_consistent();
    }

    #[test]
    #[should_panic(expected = "inactive LRU head has a predecessor")]
    fn eviction_rejects_head_with_predecessor() {
        let mut pool = VllmBlockPool::new(1);
        let mut seed = reserve(&mut pool, &[], 1).reservation;
        let id = pool.allocate_private(&mut seed);
        assert!(pool.cache_private(id, 7));
        pool.release(id);

        let (previous, _) = pool.inactive_links_mut(id);
        *previous = Some(id);
        let _ = pool.evict_one();
    }
}
