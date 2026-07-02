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
use crate::protocols::{LocalBlockHash, StorageTier};

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

/// Walk every allocated lower tier in [`lower_tier_query_order`] and build a
/// per-tier match map seeded from `device_matches`. Per-worker continuations
/// flow forward: a worker that matched N device blocks starts the host walk
/// at block N (anchored on its last device hash), and so on.
pub fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
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

        let tier_matches = indexer
            .backend()
            .query_match_details(sequence, &continuations);
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
    use crate::protocols::{LocalBlockHash, OverlapScores, WorkerWithDpRank};

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
        };

        let sequence = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let result = query_lower_tiers(&indexers, &sequence, &device_matches);
        assert!(result.is_empty());
    }
}
