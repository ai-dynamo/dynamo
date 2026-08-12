// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Normalizes chunk-keyed lower-tier removals to block-wise before batching.
//!
//! vLLM offload (lower-tier) removals are moving to *key-only*: a removal carries
//! only the chunk key (the tail block hash), not the chunk's constituent block
//! hashes. The constituents are announced only in the store event. The event
//! processor's [`super::batching`] step coalesces adjacent parent-chaining stores
//! into one multi-block store, which erases chunk boundaries before any indexer
//! sees them, so the reconstruction cannot happen downstream.
//!
//! [`ChunkNormalizer`] runs at the head of the event processor, *before* batching,
//! where each store is still one chunk. It keeps a per-`(dp_rank, storage_tier)`
//! map from a chunk's tail hash to its member block hashes and expands each
//! removal into the full block list. After expansion everything downstream is
//! block-wise: the [`super::dedup`] filter reference-counts shared blocks and the
//! lower-tier indexer stays purely block-wise.
//!
//! Expansion is a de-duplicated union, so a *fat* removal (which already lists
//! every constituent) collapses to the same block set — the normalizer is
//! therefore safe while the engine still emits fat removals, with no flag day.

use std::collections::{HashMap, HashSet};

use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, KvCacheStoreData, StorageTier};

use crate::kv_router::metrics::kv_publisher_metrics;

/// Per-worker reconstruction of block-wise removals from chunk-tail keys.
///
/// Partitioned by `(dp_rank, storage_tier)` to mirror [`super::dedup::EventDedupFilter`]:
/// the same hash on different ranks/tiers is an independent block, and the two
/// structures must be cleared in lockstep on a `Cleared` event.
#[derive(Default)]
pub(super) struct ChunkNormalizer {
    // (dp_rank, storage_tier) -> (chunk tail hash -> ordered member block hashes)
    per_key: HashMap<
        (u32, StorageTier),
        HashMap<ExternalSequenceBlockHash, Vec<ExternalSequenceBlockHash>>,
    >,
}

impl ChunkNormalizer {
    /// Record a store's chunk, keyed by its tail (last) block hash.
    ///
    /// Returns `true` if the caller should forward the store, `false` if it is a
    /// duplicate/retry of an already-tracked chunk that should be suppressed.
    /// Suppression keeps the dedup filter's unconditional per-block increment at
    /// exactly one reference per distinct chunk.
    pub(super) fn record_store(
        &mut self,
        dp_rank: u32,
        tier: StorageTier,
        data: &KvCacheStoreData,
    ) -> bool {
        let Some(tail) = data.blocks.last().map(|b| b.block_hash) else {
            // Empty/placeholder store: nothing to track, forward as-is.
            return true;
        };
        let members = self.per_key.entry((dp_rank, tier)).or_default();
        if members.contains_key(&tail) {
            if let Some(m) = kv_publisher_metrics() {
                m.add_lower_tier_normalize("duplicate_store_suppressed", 1);
            }
            return false;
        }
        members.insert(tail, data.blocks.iter().map(|b| b.block_hash).collect());
        true
    }

    /// Expand a removal's hash list to block-wise.
    ///
    /// A hash that is a tracked chunk tail is replaced by its member block hashes
    /// and dropped from the map (the chunk is being evicted); any other hash
    /// passes through unchanged. The union is de-duplicated, so a fat removal
    /// (constituents already listed) collapses to the same set.
    pub(super) fn expand_remove(
        &mut self,
        dp_rank: u32,
        tier: StorageTier,
        hashes: Vec<ExternalSequenceBlockHash>,
    ) -> Vec<ExternalSequenceBlockHash> {
        let mut out: HashSet<ExternalSequenceBlockHash> = HashSet::default();
        let mut expanded: u64 = 0;
        let mut passthrough: u64 = 0;
        match self.per_key.get_mut(&(dp_rank, tier)) {
            Some(map) => {
                for h in hashes {
                    match map.remove(&h) {
                        Some(members) => {
                            expanded += 1;
                            out.extend(members);
                        }
                        None => {
                            passthrough += 1;
                            out.insert(h);
                        }
                    }
                }
            }
            None => {
                passthrough = hashes.len() as u64;
                out.extend(hashes);
            }
        }
        if let Some(m) = kv_publisher_metrics() {
            m.add_lower_tier_normalize("tail_expanded", expanded);
            m.add_lower_tier_normalize("passthrough", passthrough);
        }
        out.into_iter().collect()
    }

    /// Drop all tracked chunks for a rank, in lockstep with the dedup filter's
    /// `clear_rank` on a `Cleared` event.
    pub(super) fn clear_rank(&mut self, dp_rank: u32) {
        self.per_key.retain(|(rank, _), _| *rank != dp_rank);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{KvCacheStoredBlockData, LocalBlockHash};

    fn h(v: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(v)
    }

    fn store(hashes: &[u64]) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|&x| KvCacheStoredBlockData {
                    block_hash: h(x),
                    tokens_hash: LocalBlockHash(x),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn sorted(mut v: Vec<ExternalSequenceBlockHash>) -> Vec<u64> {
        v.sort();
        v.into_iter().map(|x| x.0).collect()
    }

    #[test]
    fn key_only_removal_expands_to_members() {
        let mut n = ChunkNormalizer::default();
        assert!(n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3])));
        // key-only removal carries just the tail hash 3
        let out = n.expand_remove(0, StorageTier::HostPinned, vec![h(3)]);
        assert_eq!(sorted(out), vec![1, 2, 3]);
    }

    #[test]
    fn duplicate_store_is_suppressed() {
        let mut n = ChunkNormalizer::default();
        assert!(n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3])));
        assert!(!n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3])));
    }

    #[test]
    fn shared_prefix_only_expands_the_removed_chunk() {
        let mut n = ChunkNormalizer::default();
        n.record_store(0, StorageTier::HostPinned, &store(&[5, 6, 7, 8]));
        n.record_store(0, StorageTier::HostPinned, &store(&[5, 6, 9, 10]));
        // Removing chunk 8 expands to its own members only; chunk 10 keeps 5,6.
        let out = n.expand_remove(0, StorageTier::HostPinned, vec![h(8)]);
        assert_eq!(sorted(out), vec![5, 6, 7, 8]);
        let out2 = n.expand_remove(0, StorageTier::HostPinned, vec![h(10)]);
        assert_eq!(sorted(out2), vec![5, 6, 9, 10]);
    }

    #[test]
    fn fat_removal_collapses_via_union_dedup() {
        let mut n = ChunkNormalizer::default();
        n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3]));
        // fat removal already lists every constituent; tail 3 expands, 1/2 pass
        let out = n.expand_remove(0, StorageTier::HostPinned, vec![h(1), h(2), h(3)]);
        assert_eq!(sorted(out), vec![1, 2, 3]);
    }

    #[test]
    fn unknown_tail_passes_through() {
        let mut n = ChunkNormalizer::default();
        let out = n.expand_remove(0, StorageTier::HostPinned, vec![h(42)]);
        assert_eq!(sorted(out), vec![42]);
    }

    #[test]
    fn ranks_and_tiers_are_independent() {
        let mut n = ChunkNormalizer::default();
        n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3]));
        // same tail hash on a different rank is a different (untracked) chunk
        let out = n.expand_remove(1, StorageTier::HostPinned, vec![h(3)]);
        assert_eq!(sorted(out), vec![3]);
        // different tier likewise independent
        let out2 = n.expand_remove(0, StorageTier::Disk, vec![h(3)]);
        assert_eq!(sorted(out2), vec![3]);
    }

    #[test]
    fn clear_rank_drops_only_that_rank() {
        let mut n = ChunkNormalizer::default();
        n.record_store(0, StorageTier::HostPinned, &store(&[1, 2, 3]));
        n.record_store(1, StorageTier::HostPinned, &store(&[4, 5, 6]));
        n.clear_rank(0);
        assert_eq!(
            sorted(n.expand_remove(0, StorageTier::HostPinned, vec![h(3)])),
            vec![3]
        );
        assert_eq!(
            sorted(n.expand_remove(1, StorageTier::HostPinned, vec![h(6)])),
            vec![4, 5, 6]
        );
    }
}
