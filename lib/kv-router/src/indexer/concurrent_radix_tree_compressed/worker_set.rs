// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker set that keeps an order-independent fingerprint of its members.

use std::ops::Deref;

use rustc_hash::FxHashSet;

use crate::protocols::WorkerWithDpRank;

/// Workers plus the XOR of their mixed ids. Two sets with the same members have the same
/// fingerprint; two sets with different members collide with probability about 2^-64. All
/// mutation goes through this type, so the fingerprint stays in step with the members.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct WorkerSet {
    workers: FxHashSet<WorkerWithDpRank>,
    fingerprint: u64,
}

impl WorkerSet {
    pub(super) fn insert(&mut self, worker: WorkerWithDpRank) -> bool {
        let inserted = self.workers.insert(worker);
        if inserted {
            self.fingerprint ^= mix_worker(worker);
        }
        inserted
    }

    pub(super) fn remove(&mut self, worker: &WorkerWithDpRank) -> bool {
        let removed = self.workers.remove(worker);
        if removed {
            self.fingerprint ^= mix_worker(*worker);
        }
        removed
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&WorkerWithDpRank) -> bool) {
        let fingerprint = &mut self.fingerprint;
        self.workers.retain(|worker| {
            let kept = keep(worker);
            if !kept {
                *fingerprint ^= mix_worker(*worker);
            }
            kept
        });
    }

    /// Whether `other` has the same members, up to a fingerprint collision. Costs two
    /// comparisons instead of a pass over the members.
    #[inline]
    pub(super) fn same_members(&self, other: &Self) -> bool {
        self.workers.len() == other.workers.len() && self.fingerprint == other.fingerprint
    }
}

impl Deref for WorkerSet {
    type Target = FxHashSet<WorkerWithDpRank>;

    fn deref(&self) -> &Self::Target {
        &self.workers
    }
}

impl<'a> IntoIterator for &'a WorkerSet {
    type Item = &'a WorkerWithDpRank;
    type IntoIter = std::collections::hash_set::Iter<'a, WorkerWithDpRank>;

    fn into_iter(self) -> Self::IntoIter {
        self.workers.iter()
    }
}

impl FromIterator<WorkerWithDpRank> for WorkerSet {
    fn from_iter<I: IntoIterator<Item = WorkerWithDpRank>>(workers: I) -> Self {
        let mut set = Self::default();
        for worker in workers {
            set.insert(worker);
        }
        set
    }
}

#[inline]
pub(crate) fn mix_worker(worker: WorkerWithDpRank) -> u64 {
    splitmix64(splitmix64(worker.worker_id) ^ u64::from(worker.dp_rank))
}

#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64, dp_rank: u32) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, dp_rank)
    }

    #[test]
    fn fingerprint_follows_members_through_every_mutation() {
        let mut rng = fastrand::Rng::with_seed(7);
        let mut set = WorkerSet::default();
        for _ in 0..10_000 {
            let w = worker(rng.u64(..16), rng.u32(..2));
            match rng.u8(..3) {
                0 => {
                    set.insert(w);
                }
                1 => {
                    set.remove(&w);
                }
                _ => {
                    let cut = rng.u64(..16);
                    set.retain(|worker| worker.worker_id != cut);
                }
            }
            let rebuilt: WorkerSet = set.iter().copied().collect();
            assert_eq!(set.fingerprint, rebuilt.fingerprint);
            assert!(set.same_members(&rebuilt));
        }
    }

    #[test]
    fn equal_sized_sets_with_different_members_differ() {
        let a: WorkerSet = [worker(0, 0), worker(1, 0)].into_iter().collect();
        let b: WorkerSet = [worker(0, 0), worker(2, 0)].into_iter().collect();
        let c: WorkerSet = [worker(0, 0), worker(1, 1)].into_iter().collect();
        let same: WorkerSet = [worker(1, 0), worker(0, 0)].into_iter().collect();
        assert!(!a.same_members(&b));
        assert!(!a.same_members(&c));
        assert!(a.same_members(&same));
        assert!(!WorkerSet::default().same_members(&[worker(0, 0)].into_iter().collect()));
    }
}
