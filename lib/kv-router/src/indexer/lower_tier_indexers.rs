// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-tier registry of [`LowerTierIndexer`] instances and helpers for walking
//! the device → host → disk continuation chain.
//!
//! The primary KV indexer (radix tree) handles device-tier overlap scoring.
//! When a request arrives, we want to extend the per-worker match by walking
//! whichever lower tiers a worker has registered. [`LowerTierIndexers`] holds
//! one [`ThreadPoolIndexer<LowerTierIndexer>`] per non-device [`StorageTier`]
//! and lazily allocates each tier on first event arrival.
//!
//! Both the request-plane indexer (`dynamo-llm`) and the standalone HTTP
//! indexer (this crate's `services::indexer` module) share this implementation
//! so tier semantics stay aligned across the two surfaces.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::indexer::{
    KvIndexerMetrics, LowerTierContinuation, LowerTierIndexer, LowerTierMatchDetails, MatchDetails,
    ThreadPoolIndexer, WireTieredMatchDetails,
};
use crate::protocols::{LocalBlockHash, StorageTier, WorkerWithDpRank};
use crate::router_hint::RouterHintRootCandidates;
use rustc_hash::FxHashMap;

/// Holds one per-tier [`ThreadPoolIndexer<LowerTierIndexer>`] for every
/// non-device [`StorageTier`] that has received at least one event.
#[derive(Clone)]
pub struct LowerTierIndexers {
    metrics: Option<Arc<KvIndexerMetrics>>,
    num_threads: usize,
    block_size: u32,
    indexers: Arc<RwLock<HashMap<StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>>>>,
}

impl LowerTierIndexers {
    /// Metrics-less constructor for call sites without a `KvIndexerMetrics` handle.
    /// Router production assembly should use [`new_with_metrics`](Self::new_with_metrics)
    /// so lower-tier traffic is included in `kv_cache_events_applied`.
    pub fn new(num_threads: usize, block_size: u32) -> Self {
        Self::new_with_metrics(num_threads, block_size, None)
    }

    /// Same as [`new`](Self::new) but wires `kv_cache_events_applied`
    /// counters into every lazily created per-tier indexer, matching the
    /// observability of the device-tier path.
    pub fn new_with_metrics(
        num_threads: usize,
        block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> Self {
        assert!(
            num_threads > 0,
            "lower-tier indexer threads must be non-zero"
        );
        Self {
            num_threads,
            block_size,
            metrics,
            indexers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return the per-tier indexer for `storage_tier`, lazily allocating it
    /// the first time a non-device tier is seen.
    pub fn get_or_create(
        &self,
        storage_tier: StorageTier,
    ) -> Arc<ThreadPoolIndexer<LowerTierIndexer>> {
        debug_assert!(!storage_tier.is_gpu());
        if let Some(indexer) = self.indexers.read().unwrap().get(&storage_tier).cloned() {
            return indexer;
        }
        self.indexers
            .write()
            .unwrap()
            .entry(storage_tier)
            .or_insert_with(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    LowerTierIndexer::new(),
                    self.num_threads,
                    self.block_size,
                    self.metrics.clone(),
                ))
            })
            .clone()
    }

    /// All currently allocated lower-tier indexers, in unspecified order.
    pub fn all(&self) -> Vec<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().values().cloned().collect()
    }

    /// All currently allocated lower-tier indexers paired with the
    /// [`StorageTier`] each one indexes. Used by callers that need to retag
    /// per-tier dumps (e.g. peer-recovery).
    pub fn entries(&self) -> Vec<(StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>)> {
        self.indexers
            .read()
            .unwrap()
            .iter()
            .map(|(tier, indexer)| (*tier, indexer.clone()))
            .collect()
    }

    /// Lookup without allocation; returns `None` if the tier is unseen.
    pub fn get(
        &self,
        storage_tier: StorageTier,
    ) -> Option<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().get(&storage_tier).cloned()
    }
}

/// Native tiered-match container: the device-tier match plus a per-tier map
/// of lower-tier hits. Wire-friendly representations live in
/// [`WireTieredMatchDetails`]; conversions in both directions are provided.
#[derive(Debug, Clone, Default)]
pub struct TieredMatchDetails {
    pub device: MatchDetails,
    pub lower_tier: HashMap<StorageTier, LowerTierMatchDetails>,
}

impl From<&TieredMatchDetails> for WireTieredMatchDetails {
    fn from(d: &TieredMatchDetails) -> Self {
        Self {
            device: d.device.overlap_scores.clone().into(),
            lower_tier: d
                .lower_tier
                .iter()
                .map(|(tier, details)| (*tier, details.into()))
                .collect(),
        }
    }
}

impl From<WireTieredMatchDetails> for TieredMatchDetails {
    fn from(w: WireTieredMatchDetails) -> Self {
        // `last_matched_hashes` is only needed server-side to seed the tier walk,
        // so we leave it empty on the inbound side.
        let mut lower_tier = HashMap::with_capacity(w.lower_tier.len());
        for (tier, details) in w.lower_tier {
            if lower_tier.insert(tier, details.into()).is_some() {
                tracing::warn!(
                    ?tier,
                    "Duplicate StorageTier in WireTieredMatchDetails; keeping last entry"
                );
            }
        }
        Self {
            device: MatchDetails {
                overlap_scores: w.device.into(),
                ..Default::default()
            },
            lower_tier,
        }
    }
}

/// The order in which lower tiers are walked when extending a match. Device
/// → HostPinned → Disk → External.
pub fn lower_tier_query_order() -> [StorageTier; 3] {
    [
        StorageTier::HostPinned,
        StorageTier::Disk,
        StorageTier::External,
    ]
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LowerTierQueryOptions {
    pub retain_router_hint_chain: bool,
}

/// Walk every allocated lower tier in [`lower_tier_query_order`] and build a
/// per-tier match map seeded from `device_matches`. Per-worker continuations
/// flow forward: a worker that matched N device blocks starts the host walk
/// at block N (anchored on its last device hash), and so on.
pub fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    query_lower_tiers_with_options(
        indexers,
        sequence,
        device_matches,
        LowerTierQueryOptions::default(),
    )
}

fn combine_router_hint_root_candidates(
    device_candidates: Option<&RouterHintRootCandidates>,
    continuations: &FxHashMap<WorkerWithDpRank, LowerTierContinuation>,
    tier_matches: &LowerTierMatchDetails,
) -> Option<RouterHintRootCandidates> {
    let mut block_hashes = device_candidates
        .map(|candidates| candidates.block_hashes.clone())
        .unwrap_or_default();
    let mut owner_prefix_blocks: FxHashMap<WorkerWithDpRank, usize> = FxHashMap::default();

    if let Some(candidates) = device_candidates {
        owner_prefix_blocks.extend(candidates.owner_prefix_blocks.iter().copied());
    }

    let Some(extensions) = tier_matches.router_hint_extensions.as_ref() else {
        return device_candidates.cloned();
    };

    let mut extension_rows = extensions
        .iter()
        .filter_map(|(worker, hashes)| {
            let continuation = continuations.get(worker)?;
            (!hashes.is_empty()).then_some((*worker, continuation.start_pos, hashes.as_slice()))
        })
        .collect::<Vec<_>>();
    extension_rows.sort_unstable_by_key(|(worker, start_pos, _)| (*start_pos, *worker));

    for (worker, start_pos, hashes) in extension_rows {
        if block_hashes.len() < start_pos {
            continue;
        }

        let original_len = block_hashes.len();
        let mut valid = true;
        for (offset, hash) in hashes.iter().copied().enumerate() {
            let pos = start_pos + offset;
            if pos < block_hashes.len() {
                if block_hashes[pos] != hash {
                    valid = false;
                    break;
                }
            } else if pos == block_hashes.len() {
                block_hashes.push(hash);
            } else {
                valid = false;
                break;
            }
        }

        if valid {
            owner_prefix_blocks.insert(worker, start_pos + hashes.len());
        } else {
            block_hashes.truncate(original_len);
        }
    }

    let mut owner_prefix_blocks = owner_prefix_blocks
        .into_iter()
        .filter(|(_, blocks)| *blocks > 0)
        .collect::<Vec<_>>();
    if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
        return None;
    }
    owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);

    Some(RouterHintRootCandidates {
        block_hashes,
        owner_prefix_blocks,
    })
}

pub fn query_lower_tiers_with_options(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    // No lower-tier indexers are allocated, so there is no continuation
    // work to perform. Return before validating device score/hash lockstep;
    // that invariant only matters when a lower tier will consume the
    // continuations.
    if indexers.indexers.read().unwrap().is_empty() {
        return HashMap::new();
    }

    let mut continuations = LowerTierMatchDetails::default().next_continuations;
    for (worker, matched_blocks) in &device_matches.overlap_scores.scores {
        let Some(last_hash) = device_matches.last_matched_hashes.get(worker).copied() else {
            debug_assert!(
                false,
                "device match result missing last matched hash for worker {worker:?}"
            );
            continue;
        };

        continuations.insert(
            *worker,
            LowerTierContinuation::new(*matched_blocks as usize, last_hash),
        );
    }

    let mut lower_tier_matches = HashMap::new();

    for storage_tier in lower_tier_query_order() {
        let Some(indexer) = indexers.get(storage_tier) else {
            continue;
        };

        if let Some(&first_hash) = sequence.first() {
            let root_workers: Vec<_> = indexer.backend().root_workers(first_hash);
            for worker in root_workers.iter() {
                continuations
                    .entry(*worker)
                    .or_insert_with(|| LowerTierContinuation::from_root(0));
            }
        }

        let retain_router_hint_chain =
            options.retain_router_hint_chain && storage_tier == StorageTier::HostPinned;
        let mut tier_matches = indexer.backend().query_match_details_with_options(
            sequence,
            &continuations,
            retain_router_hint_chain,
        );
        if retain_router_hint_chain {
            tier_matches.router_hint_root_candidates = combine_router_hint_root_candidates(
                device_matches.router_hint_root_candidates.as_ref(),
                &continuations,
                &tier_matches,
            );
        }
        let matched_workers = tier_matches.hits.values().filter(|&&hits| hits > 0).count();
        tracing::debug!(
            ?storage_tier,
            queried_workers = continuations.len(),
            matched_workers,
            "Queried lower-tier indexer"
        );
        continuations = tier_matches.next_continuations.clone();
        lower_tier_matches.insert(storage_tier, tier_matches);
    }

    lower_tier_matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEventData, KvCacheStoreData, LocalBlockHash,
        OverlapScores, WorkerWithDpRank,
    };
    use crate::test_utils::{router_event, stored_blocks_with_sequence_hashes};

    fn local_hashes(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        parent_hash: Option<u64>,
        local_values: &[u64],
        external_hashes: &[u64],
    ) -> crate::protocols::RouterEvent {
        router_event(
            worker_id,
            event_id,
            dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(
                    &local_hashes(local_values),
                    external_hashes,
                ),
            }),
        )
    }

    #[test]
    fn query_lower_tiers_returns_empty_when_no_tiers_allocated() {
        let indexers = LowerTierIndexers::new(1, 4);

        // Mismatched device_matches: a score entry with no paired
        // `last_matched_hashes` entry. Would `debug_assert!`-panic in the
        // old body; the early-return must skip the seeding loop entirely.
        let mut overlap_scores = OverlapScores::new();
        overlap_scores
            .scores
            .insert(WorkerWithDpRank::new(99, 0), 3);
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes: Default::default(),
            router_hint_root_candidates: None,
        };

        let sequence = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let result = query_lower_tiers(&indexers, &sequence, &device_matches);
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn query_lower_tiers_extends_router_hint_chain_from_device_prefix() {
        let indexers = LowerTierIndexers::new(1, 4);
        let worker = WorkerWithDpRank::new(7, 0);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(7, 0, 0, Some(101), &[12], &[102]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let mut overlap_scores = OverlapScores::new();
        overlap_scores.scores.insert(worker, 1);
        let mut last_matched_hashes = FxHashMap::default();
        last_matched_hashes.insert(worker, ExternalSequenceBlockHash(101));
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes,
            router_hint_root_candidates: Some(RouterHintRootCandidates {
                block_hashes: vec![ExternalSequenceBlockHash(101)],
                owner_prefix_blocks: vec![(worker, 1)],
            }),
        };

        let sequence = local_hashes(&[11, 12, 13]);
        let result = query_lower_tiers_with_options(
            &indexers,
            &sequence,
            &device_matches,
            LowerTierQueryOptions {
                retain_router_hint_chain: true,
            },
        );
        let candidates = result
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.router_hint_root_candidates.as_ref())
            .unwrap();

        assert_eq!(
            candidates.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ]
        );
        assert_eq!(candidates.owner_prefix_blocks, vec![(worker, 2)]);
    }

    #[tokio::test]
    async fn query_lower_tiers_retains_router_hint_chain_when_enabled() {
        let indexers = LowerTierIndexers::new(1, 4);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(7, 0, 0, None, &[11, 12], &[101, 102]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let sequence = local_hashes(&[11, 12, 13]);
        let result = query_lower_tiers_with_options(
            &indexers,
            &sequence,
            &MatchDetails::default(),
            LowerTierQueryOptions {
                retain_router_hint_chain: true,
            },
        );
        let candidates = result
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.router_hint_root_candidates.as_ref())
            .unwrap();

        assert_eq!(
            candidates.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102)
            ]
        );
        assert_eq!(
            candidates.owner_prefix_blocks,
            vec![(WorkerWithDpRank::new(7, 0), 2)]
        );
    }
}
