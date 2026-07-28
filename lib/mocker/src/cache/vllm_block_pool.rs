// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Physical-capacity model for vLLM's GPU block pool.
//!
//! A cached hash may have several physical copies. Copy identity is internal;
//! the pool models occupancy, reference/pin state, and LRU eviction without
//! reproducing vLLM's numeric block IDs or null block.

use std::collections::BTreeMap;

use dynamo_tokens::SequenceHash;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;

new_key_type! {
    pub(crate) struct BlockCopyId;
}

#[derive(Debug)]
enum CopyState {
    Private,
    Cached {
        hash: SequenceHash,
        refs: usize,
        pins: usize,
        inactive_at: Option<u64>,
    },
}

#[derive(Debug)]
struct BlockCopy {
    state: CopyState,
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
    by_hash: FxHashMap<SequenceHash, SmallVec<[BlockCopyId; 1]>>,
    /// Ordinary LRU: lower timestamp is evicted first.
    inactive: BTreeMap<u64, BlockCopyId>,
    next_lru: u64,
    reserved: usize,
}

impl VllmBlockPool {
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            copies: SlotMap::with_key(),
            by_hash: FxHashMap::default(),
            inactive: BTreeMap::new(),
            next_lru: 0,
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

        let hits = prefix
            .iter()
            .map(|hash| {
                let Some(id) = self.first_copy(*hash) else {
                    panic!("authorized prefix hash {hash} is no longer resident")
                };
                (*hash, id)
            })
            .collect::<Vec<_>>();

        let free = self.free_capacity();
        let needed_evictions = fresh.saturating_sub(free);
        if needed_evictions > 0 {
            let inactive_hits = hits
                .iter()
                .filter_map(|(_, id)| self.is_inactive(*id).then_some(*id))
                .collect::<FxHashSet<_>>()
                .len();
            let evictable_after_pins = self.inactive.len().saturating_sub(inactive_hits);
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
        if needed_evictions > self.inactive.len() {
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
    ) -> Vec<BlockCopyId> {
        let prefix = std::mem::take(&mut reservation.prefix);
        let mut ids = Vec::with_capacity(prefix.len());
        for (hash, id) in prefix {
            self.activate_pin(id, hash);
            ids.push(id);
        }
        ids
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
        let became_visible = !self.by_hash.contains_key(&hash);
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
            inactive_at: None,
        };
        self.by_hash.entry(hash).or_default().push(id);
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
        self.copies.len() - self.inactive.len() + self.reserved
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
        self.inactive.len()
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
        self.by_hash
            .get(&hash)
            .and_then(|copies| copies.first())
            .copied()
    }

    fn is_inactive(&self, id: BlockCopyId) -> bool {
        matches!(
            &self.copies[id].state,
            CopyState::Cached {
                inactive_at: Some(_),
                ..
            }
        )
    }

    fn pin(&mut self, id: BlockCopyId) {
        let inactive_at = {
            let CopyState::Cached {
                pins, inactive_at, ..
            } = &mut self.copies[id].state
            else {
                panic!("prefix hash points to a private copy")
            };
            *pins = pins
                .checked_add(1)
                .unwrap_or_else(|| panic!("pin count overflow"));
            inactive_at.take()
        };
        if let Some(timestamp) = inactive_at {
            assert_eq!(self.inactive.remove(&timestamp), Some(id));
        }
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
        let timestamp = self.next_lru;
        self.next_lru = self
            .next_lru
            .checked_add(1)
            .unwrap_or_else(|| panic!("LRU timestamp overflow"));
        let CopyState::Cached { inactive_at, .. } = &mut self.copies[id].state else {
            panic!("private copy cannot enter inactive LRU")
        };
        assert!(inactive_at.replace(timestamp).is_none());
        assert!(self.inactive.insert(timestamp, id).is_none());
    }

    /// Evict one physical copy. A hash is returned only on its final copy.
    fn evict_one(&mut self) -> Option<SequenceHash> {
        let Some((timestamp, id)) = self.inactive.pop_first() else {
            panic!("prechecked inactive capacity disappeared")
        };
        let Some(copy) = self.copies.remove(id) else {
            panic!("inactive LRU points to a missing copy")
        };
        let CopyState::Cached {
            hash,
            refs,
            pins,
            inactive_at,
        } = copy.state
        else {
            panic!("inactive LRU points to a private copy")
        };
        assert_eq!(refs, 0);
        assert_eq!(pins, 0);
        assert_eq!(inactive_at, Some(timestamp));

        let remove_hash = {
            let Some(copies) = self.by_hash.get_mut(&hash) else {
                panic!("evicted cached hash is missing from its index")
            };
            let Some(position) = copies.iter().position(|candidate| *candidate == id) else {
                panic!("evicted copy is missing from its hash index")
            };
            copies.remove(position);
            copies.is_empty()
        };
        if remove_hash {
            self.by_hash.remove(&hash);
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
    }

    #[test]
    fn removal_is_reported_only_for_the_last_physical_copy() {
        let mut pool = VllmBlockPool::new(2);
        let mut first = reserve(&mut pool, &[], 1).reservation;
        let first_id = pool.allocate_private(&mut first);
        assert!(pool.cache_private(first_id, 3));
        let mut second = reserve(&mut pool, &[], 1).reservation;
        let second_id = pool.allocate_private(&mut second);
        assert!(!pool.cache_private(second_id, 3));
        pool.release(first_id);
        pool.release(second_id);

        let first_eviction = reserve(&mut pool, &[], 1);
        assert!(first_eviction.removed.is_empty());
        pool.cancel(first_eviction.reservation);

        let second_eviction = reserve(&mut pool, &[], 2);
        assert_eq!(second_eviction.removed, vec![3]);
        pool.cancel(second_eviction.reservation);
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
    }
}
