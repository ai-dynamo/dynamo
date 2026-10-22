// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-worker Mamba checkpoint index used to cap attention overlap.
//!
//! vLLM can reuse a cached prefix only up to the minimum over its cache groups.
//! The router indexes only the attention group, so it keeps, per worker, the
//! attention parent links and the set of hashes announced by Mamba groups. A
//! Mamba checkpoint hash equals the attention block hash at the same boundary.

use dashmap::DashMap;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::indexer::MatchDetails;
use crate::protocols::{
    CacheGroup, ExternalSequenceBlockHash, KvCacheEventData, RouterEvent, WorkerId,
    WorkerWithDpRank,
};

const ENV_MAMBA_CHECKPOINT_CAP: &str = "DYN_ROUTER_MAMBA_CHECKPOINT_CAP";

/// True when `DYN_ROUTER_MAMBA_CHECKPOINT_CAP` is `1` or `true` (case-insensitive).
pub fn mamba_checkpoint_cap_enabled() -> bool {
    std::env::var(ENV_MAMBA_CHECKPOINT_CAP)
        .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
        .unwrap_or(false)
}

/// Result of capping one worker's match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CappedMatch {
    /// Usable matched blocks. Zero when no checkpoint lies on the matched chain.
    pub blocks: u32,
    /// Checkpoint hash the usable prefix ends at. `None` when `blocks == 0`.
    pub checkpoint: Option<ExternalSequenceBlockHash>,
}

#[derive(Default)]
struct WorkerCheckpoints {
    /// Attention block hash to its parent hash.
    parents: FxHashMap<ExternalSequenceBlockHash, Option<ExternalSequenceBlockHash>>,
    /// Hashes announced by Mamba groups.
    checkpoints: FxHashSet<ExternalSequenceBlockHash>,
    /// True once the worker has emitted any Mamba event. Survives `Cleared`.
    has_mamba: bool,
}

#[derive(Default)]
pub struct MambaCheckpointIndex {
    workers: DashMap<WorkerWithDpRank, WorkerCheckpoints>,
}

impl MambaCheckpointIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a device-tier event. Mamba-tagged events update the checkpoint set;
    /// untagged events update the attention parent links.
    pub fn apply_event(&self, event: &RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut state = self.workers.entry(worker).or_default();
        match (&event.event.data, event.cache_group) {
            (KvCacheEventData::Stored(stored), Some(CacheGroup::Mamba)) => {
                state.has_mamba = true;
                for block in &stored.blocks {
                    state.checkpoints.insert(block.block_hash);
                }
            }
            (KvCacheEventData::Removed(removed), Some(CacheGroup::Mamba)) => {
                state.has_mamba = true;
                for hash in &removed.block_hashes {
                    state.checkpoints.remove(hash);
                }
            }
            (KvCacheEventData::Stored(stored), None) => {
                let mut parent = stored.parent_hash;
                for block in &stored.blocks {
                    state.parents.insert(block.block_hash, parent);
                    parent = Some(block.block_hash);
                }
            }
            (KvCacheEventData::Removed(removed), None) => {
                for hash in &removed.block_hashes {
                    state.parents.remove(hash);
                }
            }
            (KvCacheEventData::Cleared, _) => {
                state.parents.clear();
                state.checkpoints.clear();
            }
        }
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.workers
            .retain(|worker, _| worker.worker_id != worker_id);
    }

    pub fn remove_worker_dp_rank(&self, worker: WorkerWithDpRank) {
        self.workers.remove(&worker);
    }

    /// Cap a worker's matched prefix. `last_matched` is the engine hash of the
    /// last matched block and `overlap_blocks` the matched block count.
    /// Returns `None` when the worker must stay uncapped.
    pub fn cap(
        &self,
        worker: WorkerWithDpRank,
        last_matched: ExternalSequenceBlockHash,
        overlap_blocks: u32,
    ) -> Option<CappedMatch> {
        let state = self.workers.get(&worker)?;
        if !state.has_mamba {
            return None;
        }
        let mut current = last_matched;
        let mut steps = 0u32;
        loop {
            if state.checkpoints.contains(&current) {
                return Some(CappedMatch {
                    blocks: overlap_blocks - steps,
                    checkpoint: Some(current),
                });
            }
            steps += 1;
            if steps >= overlap_blocks {
                return Some(CappedMatch {
                    blocks: 0,
                    checkpoint: None,
                });
            }
            match state.parents.get(&current).copied().flatten() {
                Some(parent) => current = parent,
                None => {
                    return Some(CappedMatch {
                        blocks: 0,
                        checkpoint: None,
                    });
                }
            }
        }
    }

    /// Apply [`Self::cap`] to every worker in a primary device match. A capped
    /// worker's `last_matched_hashes` entry becomes the checkpoint hash so the
    /// lower-tier walk continues from the checkpoint.
    pub fn cap_match_details(&self, details: &mut MatchDetails) {
        let workers: Vec<WorkerWithDpRank> =
            details.overlap_scores.scores.keys().copied().collect();
        for worker in workers {
            let Some(&last_matched) = details.last_matched_hashes.get(&worker) else {
                continue;
            };
            let overlap_blocks = details.overlap_scores.scores[&worker];
            let Some(capped) = self.cap(worker, last_matched, overlap_blocks) else {
                continue;
            };
            if capped.blocks == overlap_blocks {
                continue;
            }
            tracing::debug!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                overlap_blocks,
                usable_blocks = capped.blocks,
                "Capped KV overlap at Mamba checkpoint"
            );
            match capped.checkpoint {
                Some(checkpoint) if capped.blocks > 0 => {
                    details.overlap_scores.scores.insert(worker, capped.blocks);
                    details.last_matched_hashes.insert(worker, checkpoint);
                }
                _ => {
                    details.overlap_scores.scores.remove(&worker);
                    details.last_matched_hashes.remove(&worker);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{
        KvCacheEvent, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    const W: WorkerId = 1;

    fn h(n: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(n)
    }

    fn worker() -> WorkerWithDpRank {
        WorkerWithDpRank::new(W, 0)
    }

    fn stored(parent: Option<u64>, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            W,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent.map(h),
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .map(|&n| KvCacheStoredBlockData {
                            block_hash: h(n),
                            tokens_hash: LocalBlockHash(n),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn removed(hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            W,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes.iter().map(|&n| h(n)).collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn cleared() -> RouterEvent {
        RouterEvent::new(
            W,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        )
    }

    fn mamba(event: RouterEvent) -> RouterEvent {
        event.with_cache_group(CacheGroup::Mamba)
    }

    /// Attention chain 1..=5 (hash n at depth n).
    fn index_with_chain() -> MambaCheckpointIndex {
        let index = MambaCheckpointIndex::new();
        index.apply_event(&stored(None, &[1, 2, 3, 4, 5]));
        index
    }

    #[test]
    fn worker_without_mamba_events_is_never_capped() {
        let index = index_with_chain();
        assert_eq!(index.cap(worker(), h(5), 5), None);
    }

    #[test]
    fn unknown_worker_is_never_capped() {
        let index = MambaCheckpointIndex::new();
        assert_eq!(index.cap(worker(), h(5), 5), None);
    }

    #[test]
    fn checkpoint_at_last_matched_block_keeps_full_overlap() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(4), &[5])));
        assert_eq!(
            index.cap(worker(), h(5), 5),
            Some(CappedMatch {
                blocks: 5,
                checkpoint: Some(h(5))
            })
        );
    }

    #[test]
    fn cap_walks_back_to_the_deepest_checkpoint() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(1), &[2])));
        assert_eq!(
            index.cap(worker(), h(5), 5),
            Some(CappedMatch {
                blocks: 2,
                checkpoint: Some(h(2))
            })
        );
    }

    #[test]
    fn prefers_the_deeper_of_two_checkpoints() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(1), &[2])));
        index.apply_event(&mamba(stored(Some(3), &[4])));
        assert_eq!(
            index.cap(worker(), h(5), 5),
            Some(CappedMatch {
                blocks: 4,
                checkpoint: Some(h(4))
            })
        );
    }

    #[test]
    fn no_checkpoint_on_the_chain_gives_zero() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(None, &[99])));
        assert_eq!(
            index.cap(worker(), h(5), 5),
            Some(CappedMatch {
                blocks: 0,
                checkpoint: None
            })
        );
    }

    #[test]
    fn mamba_event_without_hashes_still_marks_the_worker_hybrid() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(None, &[])));
        assert_eq!(
            index.cap(worker(), h(5), 5),
            Some(CappedMatch {
                blocks: 0,
                checkpoint: None
            })
        );
    }

    #[test]
    fn checkpoint_arriving_after_the_attention_blocks_is_found() {
        let index = MambaCheckpointIndex::new();
        index.apply_event(&stored(None, &[1, 2, 3]));
        index.apply_event(&mamba(stored(None, &[])));
        assert_eq!(index.cap(worker(), h(3), 3).unwrap().blocks, 0);
        index.apply_event(&mamba(stored(Some(2), &[3])));
        assert_eq!(index.cap(worker(), h(3), 3).unwrap().blocks, 3);
    }

    #[test]
    fn removed_checkpoint_lowers_the_cap() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(1), &[2])));
        index.apply_event(&mamba(stored(Some(4), &[5])));
        assert_eq!(index.cap(worker(), h(5), 5).unwrap().blocks, 5);
        index.apply_event(&mamba(removed(&[5])));
        assert_eq!(index.cap(worker(), h(5), 5).unwrap().blocks, 2);
    }

    #[test]
    fn removed_attention_parent_breaks_the_chain() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(1), &[2])));
        index.apply_event(&removed(&[4]));
        assert_eq!(index.cap(worker(), h(5), 5).unwrap().blocks, 0);
    }

    #[test]
    fn cleared_drops_links_and_checkpoints_but_keeps_hybrid_flag() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(4), &[5])));
        index.apply_event(&cleared());
        assert_eq!(index.cap(worker(), h(5), 5).unwrap().blocks, 0);
    }

    #[test]
    fn remove_worker_forgets_all_ranks() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(4), &[5])));
        index.remove_worker(W);
        assert_eq!(index.cap(worker(), h(5), 5), None);
    }

    #[test]
    fn cap_match_details_rewrites_score_and_last_hash() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(Some(1), &[2])));
        let mut details = MatchDetails::new();
        details.overlap_scores.scores.insert(worker(), 5);
        details.last_matched_hashes.insert(worker(), h(5));
        index.cap_match_details(&mut details);
        assert_eq!(details.overlap_scores.scores[&worker()], 2);
        assert_eq!(details.last_matched_hashes[&worker()], h(2));
    }

    #[test]
    fn cap_match_details_removes_a_worker_capped_to_zero() {
        let index = index_with_chain();
        index.apply_event(&mamba(stored(None, &[])));
        let mut details = MatchDetails::new();
        details.overlap_scores.scores.insert(worker(), 5);
        details.last_matched_hashes.insert(worker(), h(5));
        index.cap_match_details(&mut details);
        assert!(!details.overlap_scores.scores.contains_key(&worker()));
        assert!(!details.last_matched_hashes.contains_key(&worker()));
    }

    #[test]
    fn env_gate_accepts_one_and_true_only() {
        // Tests in this module do not mutate the variable concurrently with other
        // readers; the gate itself is a pure function of the variable.
        for (value, expected) in [
            ("1", true),
            ("true", true),
            ("TRUE", true),
            ("0", false),
            ("", false),
        ] {
            unsafe { std::env::set_var(ENV_MAMBA_CHECKPOINT_CAP, value) };
            assert_eq!(mamba_checkpoint_cap_enabled(), expected, "value {value:?}");
        }
        unsafe { std::env::remove_var(ENV_MAMBA_CHECKPOINT_CAP) };
        assert!(!mamba_checkpoint_cap_enabled());
    }
}
