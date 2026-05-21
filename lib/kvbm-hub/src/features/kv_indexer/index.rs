// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Position-bucketed block index.
//!
//! The index is a fixed-size `Vec<DashMap>` presized to
//! `max_seq_len / block_size`. Each bucket maps a
//! [`SequenceHash`] (a self-describing [`PositionalLineageHash`]) to the set of
//! worker `instance_id`s holding that block. The PLH carries its own
//! `position()`, so ingest needs no out-of-band position data — and queries
//! resolve by walking the candidate hashes and returning the **deepest** one
//! present (PLH lineage guarantees holders of a deep block also hold its
//! ancestors).

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use kvbm_logical::SequenceHash;
use kvbm_logical::events::{KvCacheEvents, KvbmCacheEvents};

use super::protocol::{ByPositionResponse, IndexEntry};

/// Position-bucketed map of block hash → holding instances.
pub struct PositionalIndex {
    /// `buckets[pos]` indexes blocks at sequence position `pos`. Length is
    /// `max_seq_len / block_size`.
    buckets: Vec<DashMap<SequenceHash, HashSet<u128>>>,
    block_size: usize,
    max_seq_len: usize,
    /// Count of create events whose position exceeded the last bucket.
    dropped_out_of_range: AtomicU64,
}

impl PositionalIndex {
    /// Builds an index sized for `max_seq_len` tokens at `block_size` tokens
    /// per block. Requires `block_size > 0` and `max_seq_len % block_size == 0`.
    pub fn new(max_seq_len: usize, block_size: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(block_size > 0, "block_size must be > 0");
        anyhow::ensure!(
            max_seq_len.is_multiple_of(block_size),
            "max_seq_len ({max_seq_len}) must be evenly divisible by block_size ({block_size})"
        );
        let num_positions = max_seq_len / block_size;
        let buckets = (0..num_positions).map(|_| DashMap::new()).collect();
        Ok(Self {
            buckets,
            block_size,
            max_seq_len,
            dropped_out_of_range: AtomicU64::new(0),
        })
    }

    /// Number of position buckets (`max_seq_len / block_size`).
    pub fn num_positions(&self) -> usize {
        self.buckets.len()
    }

    /// Block size (tokens per block) the index was built for.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Maximum sequence length (tokens) the index was built for.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Count of create events dropped because their position exceeded the
    /// last bucket.
    pub fn dropped_out_of_range(&self) -> u64 {
        self.dropped_out_of_range.load(Ordering::Relaxed)
    }

    /// Applies one wire batch to the index.
    pub fn apply(&self, batch: KvbmCacheEvents) {
        let instance = batch.instance_id;
        match batch.events {
            KvCacheEvents::Create(hashes) => {
                for h in hashes {
                    self.insert(h, instance);
                }
            }
            KvCacheEvents::Remove(hashes) => {
                for h in hashes {
                    self.remove(h, instance);
                }
            }
            KvCacheEvents::Shutdown => self.remove_instance(instance),
        }
    }

    fn insert(&self, hash: SequenceHash, instance: u128) {
        let pos = hash.position() as usize;
        let Some(bucket) = self.buckets.get(pos) else {
            self.dropped_out_of_range.fetch_add(1, Ordering::Relaxed);
            return;
        };
        bucket.entry(hash).or_default().insert(instance);
    }

    fn remove(&self, hash: SequenceHash, instance: u128) {
        let pos = hash.position() as usize;
        let Some(bucket) = self.buckets.get(pos) else {
            return;
        };
        // Drop the per-entry guard before `remove_if` to avoid re-locking the
        // same shard while it is held.
        let now_empty = match bucket.get_mut(&hash) {
            Some(mut set) => {
                set.remove(&instance);
                set.is_empty()
            }
            None => false,
        };
        if now_empty {
            bucket.remove_if(&hash, |_, set| set.is_empty());
        }
    }

    /// Removes `instance` from every bucket (used for `Shutdown` and
    /// registry eviction). Empty entries are pruned.
    pub fn remove_instance(&self, instance: u128) {
        for bucket in &self.buckets {
            bucket.retain(|_, set| {
                set.remove(&instance);
                !set.is_empty()
            });
        }
    }

    /// Resolves a candidate block sequence to the deepest indexed block.
    ///
    /// Returns the entry with the greatest `position()` among supplied hashes
    /// that are currently held by at least one instance. Input order does not
    /// matter.
    pub fn query(&self, hashes: &[SequenceHash]) -> Option<IndexEntry> {
        let mut best: Option<IndexEntry> = None;
        for hash in hashes {
            let pos = hash.position();
            let Some(bucket) = self.buckets.get(pos as usize) else {
                continue;
            };
            let Some(set) = bucket.get(hash) else {
                continue;
            };
            if set.is_empty() {
                continue;
            }
            if best.as_ref().is_none_or(|b| pos > b.position) {
                best = Some(entry_of(*hash, &set));
            }
        }
        best
    }

    /// Dumps the index bucket at `position`. Out-of-range positions yield an
    /// empty entry list.
    pub fn by_position(&self, position: usize) -> ByPositionResponse {
        let entries = match self.buckets.get(position) {
            Some(bucket) => bucket
                .iter()
                .map(|kv| entry_of(*kv.key(), kv.value()))
                .collect(),
            None => Vec::new(),
        };
        ByPositionResponse { position, entries }
    }
}

/// Builds a serializable [`IndexEntry`] with deterministically sorted
/// instance ids.
fn entry_of(hash: SequenceHash, instances: &HashSet<u128>) -> IndexEntry {
    let mut ids: Vec<u128> = instances.iter().copied().collect();
    ids.sort_unstable();
    IndexEntry {
        hash: format!("{hash}"),
        hash_u128: hash.as_u128().to_string(),
        position: hash.position(),
        instances: ids.into_iter().map(|i| i.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_tokens::TokenBlockSequence;
    use kvbm_logical::KvbmSequenceHashProvider;

    /// Builds `n` PLHs at positions 0..n for a given salt by laying down
    /// `n * block_size` tokens.
    fn plhs(block_size: u32, n: usize, salt: u64) -> Vec<SequenceHash> {
        let tokens: Vec<u32> = (0..(block_size as usize * n) as u32).collect();
        let seq = TokenBlockSequence::from_slice(&tokens, block_size, Some(salt));
        seq.blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect()
    }

    fn create(hashes: Vec<SequenceHash>, instance: u128) -> KvbmCacheEvents {
        KvbmCacheEvents {
            events: KvCacheEvents::Create(hashes),
            instance_id: instance,
        }
    }

    #[test]
    fn new_rejects_non_divisible() {
        assert!(PositionalIndex::new(10, 4).is_err());
        assert!(PositionalIndex::new(10, 0).is_err());
        assert!(PositionalIndex::new(16, 4).is_ok());
    }

    #[test]
    fn position_bucketing_and_by_position() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        assert_eq!(idx.num_positions(), 4);
        let hashes = plhs(4, 3, 1337);
        idx.apply(create(hashes.clone(), 100));

        // Each PLH lands in its own positional bucket.
        for (pos, h) in hashes.iter().enumerate() {
            assert_eq!(h.position() as usize, pos);
            let resp = idx.by_position(pos);
            assert_eq!(resp.entries.len(), 1);
            assert_eq!(resp.entries[0].instances, vec!["100".to_string()]);
            assert_eq!(resp.entries[0].hash_u128, h.as_u128().to_string());
        }
        // Empty / out-of-range buckets.
        assert!(idx.by_position(3).entries.is_empty());
        assert!(idx.by_position(99).entries.is_empty());
    }

    #[test]
    fn shared_prefix_lists_both_instances() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 2, 1337);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));

        let resp = idx.by_position(0);
        assert_eq!(resp.entries.len(), 1);
        assert_eq!(
            resp.entries[0].instances,
            vec!["1".to_string(), "2".to_string()]
        );
    }

    #[test]
    fn query_returns_deepest_match() {
        let idx = PositionalIndex::new(64, 4).unwrap();
        // instance 7 holds a 3-deep sequence.
        let hashes = plhs(4, 3, 42);
        idx.apply(create(hashes.clone(), 7));

        // Query with the full sequence → deepest (position 2).
        let hit = idx.query(&hashes).expect("hit");
        assert_eq!(hit.position, 2);
        assert_eq!(hit.instances, vec!["7".to_string()]);

        // Unsorted input still yields the deepest present.
        let mut shuffled = hashes.clone();
        shuffled.reverse();
        assert_eq!(idx.query(&shuffled).unwrap().position, 2);

        // A query whose deep blocks are unknown falls back to the shallow hit.
        let unknown = plhs(4, 5, 999);
        let mut mixed = vec![hashes[0], hashes[1]];
        mixed.extend_from_slice(&unknown[2..]); // positions 2..4 unknown
        let hit = idx.query(&mixed).expect("shallow hit");
        assert_eq!(hit.position, 1);
    }

    #[test]
    fn query_miss_returns_none() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 2, 1);
        assert!(idx.query(&hashes).is_none());
    }

    #[test]
    fn remove_prunes_entry_when_last_holder_leaves() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 1, 5);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));

        idx.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 1,
        });
        assert_eq!(
            idx.by_position(0).entries[0].instances,
            vec!["2".to_string()]
        );

        idx.apply(KvbmCacheEvents {
            events: KvCacheEvents::Remove(hashes.clone()),
            instance_id: 2,
        });
        assert!(idx.by_position(0).entries.is_empty());
    }

    #[test]
    fn remove_instance_sweeps_all_positions() {
        let idx = PositionalIndex::new(16, 4).unwrap();
        let hashes = plhs(4, 3, 5);
        idx.apply(create(hashes.clone(), 1));
        idx.apply(create(hashes.clone(), 2));
        idx.remove_instance(1);
        for pos in 0..3 {
            assert_eq!(
                idx.by_position(pos).entries[0].instances,
                vec!["2".to_string()]
            );
        }
        idx.remove_instance(2);
        for pos in 0..3 {
            assert!(idx.by_position(pos).entries.is_empty());
        }
    }

    #[test]
    fn out_of_range_create_is_dropped_with_counter() {
        let idx = PositionalIndex::new(8, 4).unwrap(); // 2 positions: 0,1
        let hashes = plhs(4, 4, 9); // positions 0..3
        idx.apply(create(hashes, 1));
        assert_eq!(idx.dropped_out_of_range(), 2); // positions 2,3 dropped
        assert_eq!(idx.by_position(0).entries.len(), 1);
        assert_eq!(idx.by_position(1).entries.len(), 1);
    }
}
