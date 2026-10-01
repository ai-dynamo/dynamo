// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sparse position-bucketed index of positional lineage hashes (PLHs).
//!
//! Each PLH carries its own position, so ingestion needs no out-of-band position.
//! Entries can be sparse: carriers may sit at scattered positions without requiring a
//! parent chain, and queries return the deepest held hash. The holder type is generic
//! (for example, KVBM hub instance IDs or router `WorkerWithDpRank` values). Manifest
//! scoping, event decoding, and create-kind policy belong to callers.

use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use dynamo_tokens::PositionalLineageHash;

type Bucket<W> = Arc<DashMap<PositionalLineageHash, FxHashSet<W>, FxBuildHasher>>;

/// Sparse index of PLHs held by generic identifiers.
pub struct LineageIndex<W> {
    buckets: DashMap<u64, Bucket<W>, FxBuildHasher>,
    max_positions: AtomicU64,
    dropped_out_of_range: AtomicU64,
    swap: parking_lot::RwLock<()>,
}

/// A PLH and its sorted set of holders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageHit<W> {
    /// The matched positional lineage hash.
    pub hash: PositionalLineageHash,
    /// Holder identifiers in ascending order.
    pub holders: Vec<W>,
}

impl<W: Copy + Eq + Hash + Ord> LineageIndex<W> {
    /// Creates an index with capacity for `max_positions` positions.
    pub fn new(max_positions: u64) -> Self {
        Self {
            buckets: DashMap::with_hasher(FxBuildHasher::default()),
            max_positions: AtomicU64::new(max_positions),
            dropped_out_of_range: AtomicU64::new(0),
            swap: parking_lot::RwLock::new(()),
        }
    }

    /// Returns the current position capacity.
    pub fn max_positions(&self) -> u64 {
        self.max_positions.load(Ordering::Relaxed)
    }

    /// Raises the position capacity without lowering it.
    pub fn grow_to(&self, max_positions: u64) {
        self.max_positions
            .fetch_max(max_positions, Ordering::Relaxed);
    }

    /// Returns the number of hashes skipped because they exceeded capacity.
    pub fn dropped_out_of_range(&self) -> u64 {
        self.dropped_out_of_range.load(Ordering::Relaxed)
    }

    /// Adds `holder` for each in-range hash.
    pub fn insert(&self, holder: W, hashes: &[PositionalLineageHash]) {
        let _swap = self.swap.read();
        self.insert_unlocked(holder, hashes);
    }

    /// Removes `holder` from the specified hashes.
    pub fn remove(&self, holder: W, hashes: &[PositionalLineageHash]) {
        let _swap = self.swap.read();
        self.remove_unlocked(holder, hashes);
    }

    /// Removes `holder` from every position.
    pub fn remove_holder(&self, holder: W) {
        let _swap = self.swap.write();
        self.remove_holder_unlocked(holder);
    }

    /// Replaces `holder`'s hashes atomically with respect to queries.
    pub fn replace_holder(&self, holder: W, hashes: &[PositionalLineageHash]) {
        let _swap = self.swap.write();
        self.remove_holder_unlocked(holder);
        self.insert_unlocked(holder, hashes);
    }

    /// Returns the deepest supplied hash currently held by any holder.
    pub fn deepest(&self, hashes: &[PositionalLineageHash]) -> Option<LineageHit<W>> {
        let _swap = self.swap.read();
        let mut best: Option<LineageHit<W>> = None;

        for hash in hashes {
            let position = hash.position();
            let Some(bucket) = self.bucket(position) else {
                continue;
            };
            let Some(holders) = bucket.get(hash) else {
                continue;
            };
            if holders.is_empty() {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|hit| position > hit.hash.position())
            {
                let mut holders: Vec<W> = holders.iter().copied().collect();
                holders.sort_unstable();
                best = Some(LineageHit { hash: *hash, holders });
            }
        }

        best
    }

    /// Returns each matching holder's deepest supplied hash, sorted by holder.
    pub fn deepest_by_holder(
        &self,
        hashes: &[PositionalLineageHash],
    ) -> Vec<(W, PositionalLineageHash)> {
        let _swap = self.swap.read();
        let mut best = FxHashMap::<W, (u64, PositionalLineageHash)>::default();

        for hash in hashes {
            let position = hash.position();
            let Some(bucket) = self.bucket(position) else {
                continue;
            };
            let Some(holders) = bucket.get(hash) else {
                continue;
            };
            for holder in holders.iter().copied() {
                best.entry(holder)
                    .and_modify(|(best_position, best_hash)| {
                        if position > *best_position {
                            *best_position = position;
                            *best_hash = *hash;
                        }
                    })
                    .or_insert((position, *hash));
            }
        }

        let mut result: Vec<_> = best
            .into_iter()
            .map(|(holder, (_, hash))| (holder, hash))
            .collect();
        result.sort_unstable_by_key(|(holder, _)| *holder);
        result
    }

    /// Returns all entries at `position`, sorted by hash and holder.
    pub fn entries_at(&self, position: u64) -> Vec<LineageHit<W>> {
        let _swap = self.swap.read();
        let Some(bucket) = self.bucket(position) else {
            return Vec::new();
        };

        let mut entries: Vec<_> = bucket
            .iter()
            .map(|entry| {
                let mut holders: Vec<W> = entry.value().iter().copied().collect();
                holders.sort_unstable();
                LineageHit {
                    hash: *entry.key(),
                    holders,
                }
            })
            .collect();
        entries.sort_unstable_by_key(|entry| entry.hash.as_u128());
        entries
    }

    fn insert_unlocked(&self, holder: W, hashes: &[PositionalLineageHash]) {
        for hash in hashes {
            let position = hash.position();
            if position >= self.max_positions.load(Ordering::Relaxed) {
                self.dropped_out_of_range.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let bucket = {
                let entry = self.buckets.entry(position).or_insert_with(|| {
                    Arc::new(DashMap::with_hasher(FxBuildHasher::default()))
                });
                Arc::clone(entry.value())
            };
            bucket.entry(*hash).or_default().insert(holder);
        }
    }

    fn remove_unlocked(&self, holder: W, hashes: &[PositionalLineageHash]) {
        for hash in hashes {
            let Some(bucket) = self.bucket(hash.position()) else {
                continue;
            };

            let now_empty = match bucket.get_mut(hash) {
                Some(mut holders) => {
                    holders.remove(&holder);
                    holders.is_empty()
                }
                None => false,
            };
            if now_empty {
                bucket.remove_if(hash, |_, holders| holders.is_empty());
            }
        }
    }

    fn remove_holder_unlocked(&self, holder: W) {
        let buckets: Vec<_> = self
            .buckets
            .iter()
            .map(|bucket| Arc::clone(bucket.value()))
            .collect();
        for bucket in buckets {
            bucket.retain(|_, holders| {
                holders.remove(&holder);
                !holders.is_empty()
            });
        }
    }

    fn bucket(&self, position: u64) -> Option<Bucket<W>> {
        self.buckets
            .get(&position)
            .map(|bucket| Arc::clone(bucket.value()))
    }
}

#[cfg(test)]
mod tests {
    use super::{LineageHit, LineageIndex};
    use crate::protocols::WorkerWithDpRank;
    use dynamo_tokens::PositionalLineageHash;

    const BLOCK_SIZE: u32 = 4;

    fn plhs(tokens: Vec<u32>) -> anyhow::Result<Vec<PositionalLineageHash>> {
        Ok(dynamo_kv_hashing::Request::builder()
            .tokens(tokens)
            .build()?
            .positional_lineage_hashes(BLOCK_SIZE)?)
    }

    fn sequence(start: u32, blocks: u32) -> anyhow::Result<Vec<PositionalLineageHash>> {
        plhs((start..start + blocks * BLOCK_SIZE).collect())
    }

    #[test]
    fn entries_at_buckets_positions_and_sorts_shared_prefix_holders() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let first = sequence(0, 4)?;
        let second = plhs(vec![
            0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        ])?;
        assert_eq!(first[0], second[0]);
        assert_ne!(first[1], second[1]);

        index.insert(9, &first);
        index.insert(2, &second);

        assert_eq!(
            index.entries_at(0),
            vec![LineageHit {
                hash: first[0],
                holders: vec![2, 9],
            }]
        );

        let mut expected = vec![
            LineageHit {
                hash: first[1],
                holders: vec![9],
            },
            LineageHit {
                hash: second[1],
                holders: vec![2],
            },
        ];
        expected.sort_unstable_by_key(|entry| entry.hash.as_u128());
        assert_eq!(index.entries_at(1), expected);
        assert!(index.entries_at(4).is_empty());
        Ok(())
    }

    #[test]
    fn deepest_selects_the_deepest_hash_regardless_of_input_order() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let hashes = sequence(10, 4)?;
        index.insert(1, &hashes);

        assert_eq!(
            index.deepest(&hashes).map(|hit| hit.hash),
            Some(hashes[3])
        );
        let mut reversed = hashes.clone();
        reversed.reverse();
        assert_eq!(
            index.deepest(&reversed).map(|hit| hit.hash),
            Some(hashes[3])
        );
        Ok(())
    }

    #[test]
    fn deepest_miss_returns_none_and_holders_are_sorted() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let hashes = sequence(20, 3)?;
        let other = sequence(80, 3)?;
        index.insert(9, &hashes);
        index.insert(2, &hashes);

        assert_eq!(
            index.deepest(&[hashes[1]]),
            Some(LineageHit {
                hash: hashes[1],
                holders: vec![2, 9],
            })
        );
        assert!(index.deepest(&other).is_none());
        Ok(())
    }

    #[test]
    fn deepest_ties_choose_the_first_hash_in_input_order() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let first = sequence(0, 3)?;
        let second = plhs(vec![0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47])?;
        assert_ne!(first[1], second[1]);
        index.insert(1, &[first[1]]);
        index.insert(2, &[second[1]]);

        assert_eq!(
            index.deepest(&[first[1], second[1]]).map(|hit| hit.hash),
            Some(first[1])
        );
        assert_eq!(
            index.deepest(&[second[1], first[1]]).map(|hit| hit.hash),
            Some(second[1])
        );
        Ok(())
    }

    #[test]
    fn remove_prunes_empty_hash_entries_and_preserves_remaining_holders() -> anyhow::Result<()> {
        let index = LineageIndex::new(3);
        let hashes = sequence(0, 3)?;
        index.insert(1, &hashes);
        index.insert(2, &hashes);

        index.remove(1, &[hashes[0]]);
        assert_eq!(index.entries_at(0)[0].holders, vec![2]);
        index.remove(2, &[hashes[0]]);
        assert!(index.entries_at(0).is_empty());
        assert_eq!(index.entries_at(1)[0].holders, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn remove_holder_sweeps_all_positions() -> anyhow::Result<()> {
        let index = LineageIndex::new(3);
        let hashes = sequence(0, 3)?;
        index.insert(1, &hashes);
        index.insert(2, &hashes);

        index.remove_holder(1);
        for position in 0..3 {
            assert_eq!(index.entries_at(position)[0].holders, vec![2]);
        }
        index.remove_holder(2);
        for position in 0..3 {
            assert!(index.entries_at(position).is_empty());
        }
        Ok(())
    }

    #[test]
    fn capacity_drops_out_of_range_hashes_and_grows_only() -> anyhow::Result<()> {
        let index = LineageIndex::new(2);
        let hashes = sequence(0, 4)?;
        index.insert(1, &hashes);

        assert_eq!(index.dropped_out_of_range(), 2);
        assert!(index.entries_at(2).is_empty());

        index.grow_to(4);
        index.grow_to(1);
        assert_eq!(index.max_positions(), 4);
        index.insert(1, &hashes[2..]);

        assert_eq!(index.dropped_out_of_range(), 2);
        assert_eq!(index.entries_at(2)[0].holders, vec![1]);
        assert_eq!(index.entries_at(3)[0].holders, vec![1]);
        Ok(())
    }

    #[test]
    fn replace_holder_replaces_only_that_holders_hashes() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let old_hashes = sequence(0, 4)?;
        let new_hashes = sequence(100, 4)?;
        index.insert(1, &old_hashes);
        index.insert(2, &old_hashes);

        index.replace_holder(1, &new_hashes);

        assert_eq!(
            index.deepest_by_holder(&old_hashes),
            vec![(2, old_hashes[3])]
        );
        assert_eq!(
            index.deepest_by_holder(&new_hashes),
            vec![(1, new_hashes[3])]
        );
        Ok(())
    }

    #[test]
    fn deepest_by_holder_supports_sparse_holdings_and_excludes_other_chains() -> anyhow::Result<()> {
        let index = LineageIndex::new(4);
        let hashes = sequence(0, 4)?;
        let other = sequence(100, 4)?;
        index.insert(1, &[hashes[1]]);
        index.insert(2, &[hashes[3]]);
        index.insert(3, &[other[2]]);

        assert_eq!(
            index.deepest_by_holder(&hashes),
            vec![(1, hashes[1]), (2, hashes[3])]
        );
        Ok(())
    }

    #[test]
    fn supports_worker_with_dp_rank_holders() -> anyhow::Result<()> {
        let index = LineageIndex::new(2);
        let hashes = sequence(0, 2)?;
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(3, 1);
        index.insert(first, &hashes);
        index.insert(second, &hashes);

        assert_eq!(
            index.deepest(&hashes).map(|hit| hit.holders),
            Some(vec![second, first])
        );
        assert_eq!(
            index.deepest_by_holder(&hashes),
            vec![(second, hashes[1]), (first, hashes[1])]
        );
        Ok(())
    }
}
