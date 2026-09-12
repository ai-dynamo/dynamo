// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Match lookups over an [`Indexer`]: primary device tier, lower tiers seeded
//! from confirmed primary matches, and the predict-on-route side overlay.

use crate::indexer::{
    KvIndexer, KvRouterError, LowerTierIndexers, LowerTierQueryOptions, MatchDetails,
    TieredMatchDetails, query_lower_tiers_with_options,
};
use crate::protocols::{LocalBlockHash, OverlapScores};
use crate::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};

use super::backend::{Indexer, RemotePrimary, SideIndexer};

/// Block hashes for one lookup, borrowed when the caller keeps them and owned
/// when it does not, so a borrowed query is copied only at the one boundary
/// that needs an owned `Vec`.
pub enum HashInput<'a> {
    Borrowed(&'a [LocalBlockHash]),
    Owned(Vec<LocalBlockHash>),
}

impl<'a> HashInput<'a> {
    pub fn as_slice(&self) -> &[LocalBlockHash] {
        match self {
            Self::Borrowed(sequence) => sequence,
            Self::Owned(sequence) => sequence,
        }
    }

    fn clone_for_boundary(&self) -> Vec<LocalBlockHash> {
        self.as_slice().to_vec()
    }

    pub(super) fn into_owned_at_boundary(self) -> Vec<LocalBlockHash> {
        match self {
            Self::Borrowed(sequence) => sequence.to_vec(),
            Self::Owned(sequence) => sequence,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct LookupPipeline<'a> {
    primary: PrimaryLookup<'a>,
    lower_tier: Option<&'a LowerTierIndexers>,
    side: Option<&'a SideIndexer>,
}

#[derive(Clone, Copy)]
enum PrimaryLookup<'a> {
    KvIndexer(&'a KvIndexer),
    Concurrent(&'a ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
    Remote(&'a dyn RemotePrimary),
    None,
}

impl Indexer {
    pub(super) fn lookup_pipeline(&self) -> LookupPipeline<'_> {
        match self {
            Self::Single {
                primary,
                lower_tier,
                approx,
                ..
            } => LookupPipeline {
                primary: PrimaryLookup::KvIndexer(primary),
                lower_tier: Some(lower_tier),
                side: approx.as_ref(),
            },
            Self::Concurrent {
                primary,
                lower_tier,
                approx,
                ..
            } => LookupPipeline {
                primary: PrimaryLookup::Concurrent(primary.as_ref()),
                lower_tier: Some(lower_tier),
                side: approx.as_ref(),
            },
            Self::Remote {
                primary, approx, ..
            } => LookupPipeline {
                primary: PrimaryLookup::Remote(primary.as_ref()),
                lower_tier: None,
                side: approx.as_ref(),
            },
            Self::None => LookupPipeline {
                primary: PrimaryLookup::None,
                lower_tier: None,
                side: None,
            },
        }
    }

    /// Device-tier match details with the side overlay merged in.
    pub async fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_match_details(HashInput::Owned(sequence))
            .await
    }

    /// Device-tier match details from the primary only (no side overlay).
    pub async fn find_primary_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_match_details(HashInput::Owned(sequence))
            .await
    }

    /// Device match details plus per-tier hits. Router-hint chain retention is
    /// honored as requested; see [`Indexer::find_tiered_matches_with_options`]
    /// for the capability-gated variant.
    pub async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_matches_by_tier_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    pub async fn find_matches_by_tier_with_options(
        &self,
        sequence: Vec<LocalBlockHash>,
        lower_tier_options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_matches_by_tier(HashInput::Owned(sequence), lower_tier_options)
            .await
    }

    pub async fn find_matches_by_tier_ref(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_matches_by_tier_ref_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    pub async fn find_matches_by_tier_ref_with_options(
        &self,
        sequence: &[LocalBlockHash],
        lower_tier_options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_matches_by_tier(HashInput::Borrowed(sequence), lower_tier_options)
            .await
    }

    /// Tiered matches from the primary and lower tiers only (no side overlay).
    pub async fn find_primary_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_matches_by_tier(HashInput::Owned(sequence))
            .await
    }
}

impl<'a> LookupPipeline<'a> {
    async fn find_match_details(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        if self.side.is_none() {
            return self.find_primary_match_details(sequence).await;
        }

        let primary_details = self.primary.find_match_details_retained(&sequence).await?;
        Ok(merge_side_or_warn(self.side, primary_details, sequence).await)
    }

    async fn find_primary_match_details(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.primary.find_match_details(sequence).await
    }

    async fn find_matches_by_tier(
        &self,
        sequence: HashInput<'_>,
        lower_tier_options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self.primary {
            PrimaryLookup::KvIndexer(_) | PrimaryLookup::Concurrent(_) => {
                let Some(lower_tier) = self.lower_tier else {
                    return Ok(TieredMatchDetails::default());
                };
                // Seed lower-tier continuations from confirmed primary matches
                // only. Predict-on-route side scores are unconfirmed; using
                // them as lower-tier anchors would over-credit host/disk cache
                // hits and break the score/hash lockstep `query_lower_tiers`
                // expects.
                let primary_device = self
                    .primary
                    .find_match_details_retained_with_options(
                        &sequence,
                        lower_tier_options.retain_kv_transfer_chain,
                    )
                    .await?;
                let lt = query_lower_tiers_with_options(
                    lower_tier,
                    sequence.as_slice(),
                    &primary_device,
                    lower_tier_options,
                );
                let device = merge_side_or_warn(self.side, primary_device, sequence).await;

                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            PrimaryLookup::Remote(primary) => {
                if lower_tier_options.retain_kv_transfer_chain {
                    tracing::warn!(
                        "KV transfer chain retention is not supported with remote primary indexer; proceeding without KV transfer hints"
                    );
                }
                let Some(side) = self.side else {
                    return primary
                        .find_matches_by_tier(sequence.into_owned_at_boundary(), false)
                        .await
                        .map_err(|e| {
                            tracing::warn!(error = %e, "Remote indexer tiered query failed");
                            KvRouterError::IndexerOffline
                        });
                };
                let mut tiered = primary
                    .find_matches_by_tier(sequence.clone_for_boundary(), false)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Remote indexer tiered query failed");
                        KvRouterError::IndexerOffline
                    })?;
                tiered.device = merge_side_or_warn(Some(side), tiered.device, sequence).await;
                Ok(tiered)
            }
            PrimaryLookup::None => Ok(TieredMatchDetails::default()),
        }
    }

    async fn find_primary_matches_by_tier(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self.primary {
            PrimaryLookup::KvIndexer(_) | PrimaryLookup::Concurrent(_) => {
                let Some(lower_tier) = self.lower_tier else {
                    return Ok(TieredMatchDetails::default());
                };
                let device = self.primary.find_match_details_retained(&sequence).await?;
                let lt = query_lower_tiers_with_options(
                    lower_tier,
                    sequence.as_slice(),
                    &device,
                    LowerTierQueryOptions::default(),
                );
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            PrimaryLookup::Remote(primary) => primary
                .find_matches_by_tier(sequence.into_owned_at_boundary(), false)
                .await
                .map_err(|e| {
                    tracing::warn!(error = %e, "Remote indexer tiered query failed");
                    KvRouterError::IndexerOffline
                }),
            PrimaryLookup::None => Ok(TieredMatchDetails::default()),
        }
    }
}

impl<'a> PrimaryLookup<'a> {
    async fn find_match_details(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = match self {
            Self::KvIndexer(primary) => {
                primary
                    .find_match_details(sequence.into_owned_at_boundary())
                    .await?
            }
            Self::Concurrent(primary) => primary
                .backend()
                .find_match_details_impl(sequence.as_slice(), false),
            Self::Remote(primary) => {
                let tiered = primary
                    .find_matches_by_tier(sequence.into_owned_at_boundary(), true)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Remote indexer query failed");
                        KvRouterError::IndexerOffline
                    })?;
                tiered.device
            }
            Self::None => return Ok(MatchDetails::new()),
        };

        Ok(primary_details)
    }

    async fn find_match_details_retained(
        &self,
        sequence: &HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.find_match_details_retained_with_options(sequence, false)
            .await
    }

    async fn find_match_details_retained_with_options(
        &self,
        sequence: &HashInput<'_>,
        retain_kv_transfer_chain: bool,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = match self {
            Self::KvIndexer(primary) => {
                primary
                    .find_match_details_with_options(
                        sequence.clone_for_boundary(),
                        retain_kv_transfer_chain,
                    )
                    .await?
            }
            Self::Concurrent(primary) => primary.backend().find_match_details_impl_with_options(
                sequence.as_slice(),
                false,
                retain_kv_transfer_chain,
            ),
            Self::Remote(primary) => {
                let tiered = primary
                    .find_matches_by_tier(sequence.clone_for_boundary(), true)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Remote indexer query failed");
                        KvRouterError::IndexerOffline
                    })?;
                tiered.device
            }
            Self::None => return Ok(MatchDetails::new()),
        };

        Ok(primary_details)
    }
}

/// Merge a side-indexer's `OverlapScores` into the primary's `MatchDetails`
/// by taking the per-worker max overlap. The side indexer covers the window
/// before the engine's first KV event arrives; for workers it knows about,
/// we use whichever indexer saw the longer prefix. `last_matched_hashes`,
/// `frequencies`, and `tree_sizes` come from the primary -- the side
/// indexer's short-TTL view isn't meaningful for those signals.
///
/// IMPORTANT: the returned `MatchDetails` is no longer guaranteed to satisfy
/// `overlap_scores.scores` <-> `last_matched_hashes` lockstep. Side-only
/// workers gain a score with no paired hash by design. The result is safe
/// for scheduling / cache-hit signal but MUST NOT be used to seed
/// `query_lower_tiers`, which assumes the lockstep invariant. The local
/// arm of `find_matches_by_tier` enforces this by running the lower-tier
/// query against primary-only `MatchDetails` before merging side scores.
fn merge_overlap_scores(mut primary: MatchDetails, side: OverlapScores) -> MatchDetails {
    for (worker, side_score) in side.scores {
        primary
            .overlap_scores
            .scores
            .entry(worker)
            .and_modify(|s| {
                if side_score > *s {
                    *s = side_score;
                }
            })
            .or_insert(side_score);
    }
    primary
}

/// Query the predict-on-route side indexer (if present) and merge its scores
/// into the primary device match details. Side scores never feed lower-tier
/// or shared-cache scoring. On query error, log a warning and return `primary`
/// unchanged so the caller still has a usable scheduling signal. See
/// [`merge_overlap_scores`] for the lockstep caveat on the returned shape.
///
/// NOTE: when this merged `MatchDetails` is combined with lower-tier hits
/// seeded from the primary-only anchor (e.g. in `find_matches_by_tier`), the
/// total cached-token signal can in theory overcount: the device score is
/// raised by the side indexer but the lower-tier walk used the lower primary
/// depth. Accepted as edge for now since side scores are short-TTL
/// approximations and the overcount is bounded and rare in practice.
pub(super) async fn merge_side_or_warn(
    side: Option<&SideIndexer>,
    primary: MatchDetails,
    sequence: HashInput<'_>,
) -> MatchDetails {
    let Some(side) = side else {
        return primary;
    };
    match side.find_matches_input(sequence).await {
        Ok(side_scores) => merge_overlap_scores(primary, side_scores),
        Err(error) => {
            tracing::warn!(
                error = %error,
                "predict-on-route side indexer query failed; using primary only"
            );
            primary
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use crate::indexer::LowerTierIndexers;
    use crate::services::indexer::backend::Indexer;
    use crate::services::indexer::backend::test_util::store_event;
    use crate::{
        ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
        approx::PruneConfig,
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, RoutingDecisionHashes},
        protocols::{
            BlockHashOptions, LocalBlockHash, StorageTier, TokensWithHashes, WorkerWithDpRank,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
    };

    fn make_test_indexer() -> Indexer {
        Indexer::Single {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
            primary_records_routing_decisions: false,
        }
    }

    fn make_test_concurrent_indexer() -> Indexer {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                2,
                4,
            )),
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: None,
            primary_records_routing_decisions: false,
        }
    }

    fn make_test_concurrent_approx_indexer() -> Indexer {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new_with_pruning(
                ConcurrentRadixTreeCompressed::new(),
                2,
                4,
                PruneConfig {
                    ttl: Duration::from_secs(60),
                },
            )),
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: None,
            primary_records_routing_decisions: true,
        }
    }

    #[test]
    fn overlap_refresh_is_limited_to_local_indexers() {
        assert!(make_test_indexer().supports_overlap_refresh());
        assert!(make_test_concurrent_indexer().supports_overlap_refresh());
        assert!(!Indexer::None.supports_overlap_refresh());
    }

    #[test]
    fn router_hint_chain_retention_requires_event_driven_primary() {
        assert!(make_test_indexer().supports_kv_transfer_chain_retention());
        assert!(make_test_concurrent_indexer().supports_kv_transfer_chain_retention());
        assert!(!make_test_concurrent_approx_indexer().supports_kv_transfer_chain_retention());
        assert!(!Indexer::None.supports_kv_transfer_chain_retention());
    }

    async fn flush_indexer(indexer: &Indexer) {
        match indexer {
            Indexer::Single {
                primary,
                lower_tier,
                ..
            } => {
                let _ = primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } => {
                primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
            }
            Indexer::Remote { .. } | Indexer::None => {}
        }
    }

    async fn assert_rank_reset_is_acknowledged(indexer: Indexer) {
        let reset_rank = WorkerWithDpRank::new(7, 0);
        let retained_rank = WorkerWithDpRank::new(7, 1);

        for dp_rank in [reset_rank.dp_rank, retained_rank.dp_rank] {
            indexer
                .try_apply_event(store_event(7, dp_rank, 1, &[], &[41], StorageTier::Device))
                .await
                .unwrap();
            indexer
                .try_apply_event(store_event(
                    7,
                    dp_rank,
                    2,
                    &[41],
                    &[42],
                    StorageTier::HostPinned,
                ))
                .await
                .unwrap();
        }
        flush_indexer(&indexer).await;

        indexer
            .reset_worker_dp_rank_and_wait(reset_rank.worker_id, reset_rank.dp_rank)
            .await
            .unwrap();

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(41), LocalBlockHash(42)])
            .await
            .unwrap();
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&reset_rank)
        );
        assert_eq!(
            matches.device.overlap_scores.scores.get(&retained_rank),
            Some(&1)
        );
        let host_hits = &matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .unwrap()
            .hits;
        assert!(!host_hits.contains_key(&reset_rank));
        assert_eq!(host_hits.get(&retained_rank), Some(&1));
    }

    #[tokio::test]
    async fn single_thread_rank_reset_waits_for_all_local_tiers() {
        assert_rank_reset_is_acknowledged(make_test_indexer()).await;
    }

    #[tokio::test]
    async fn concurrent_rank_reset_waits_for_all_local_tiers() {
        assert_rank_reset_is_acknowledged(make_test_concurrent_indexer()).await;
    }

    #[tokio::test]
    async fn tiered_query_chains_device_host_and_disk() {
        let indexer = make_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .try_apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(
                7,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
                LocalBlockHash(14),
            ])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn router_dump_includes_all_allocated_physical_tiers() {
        let indexer = make_test_indexer();
        for (event_id, tier, block) in [
            (1, StorageTier::Device, 11),
            (2, StorageTier::HostPinned, 12),
            (3, StorageTier::Disk, 13),
        ] {
            indexer
                .try_apply_event(store_event(7, 0, event_id, &[], &[block], tier))
                .await
                .unwrap();
        }
        flush_indexer(&indexer).await;

        let events = indexer.dump_events().await.unwrap();
        for tier in [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::Disk,
        ] {
            assert!(events.iter().any(|event| event.storage_tier == tier));
        }
    }

    #[tokio::test]
    async fn tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores() {
        let indexer = make_test_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .try_apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(21)])
            .await
            .unwrap();

        assert_eq!(
            matches.device.overlap_scores.scores.get(&device_worker),
            Some(&1)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&host_only_worker)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&disk_only_worker)
        );

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&host_only_worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&disk_only_worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn tiered_query_only_seeds_matching_root_workers() {
        let indexer = make_test_indexer();
        let matching_host_worker = WorkerWithDpRank::new(20, 0);
        let nonmatching_host_worker = WorkerWithDpRank::new(21, 0);

        indexer
            .try_apply_event(store_event(20, 0, 1, &[], &[31], StorageTier::HostPinned))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(21, 0, 2, &[], &[32], StorageTier::HostPinned))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(31)])
            .await
            .unwrap();

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&matching_host_worker)),
            Some(&1)
        );
        assert!(
            !matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .is_some_and(|tier| tier.hits.contains_key(&nonmatching_host_worker))
        );
    }

    #[tokio::test]
    async fn concurrent_tiered_query_chains_device_and_lower_tier_matches() {
        let indexer = make_test_concurrent_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .try_apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
            ])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn concurrent_records_hashed_routing_decision() {
        let indexer = make_test_concurrent_approx_indexer();
        assert!(indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let tokens = vec![1, 2, 3, 4];
        let block_hashes = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);

        indexer
            .record_hashed_routing_decision(worker, block_hashes.clone(), sequence_hashes)
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(block_hashes).await.unwrap();
        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&1));
    }

    #[tokio::test]
    async fn concurrent_records_precomputed_routing_hashes() {
        let indexer = make_test_concurrent_approx_indexer();
        assert!(indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let local_hashes = vec![LocalBlockHash(91), LocalBlockHash(92)];
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        indexer
            .record_routing_decision_hashes(
                worker,
                RoutingDecisionHashes {
                    local_hashes: local_hashes.clone(),
                    sequence_hashes,
                },
            )
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(local_hashes).await.unwrap();
        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&2));
    }

    #[tokio::test]
    async fn event_driven_primary_without_side_skips_route_recording() {
        let indexer = make_test_indexer();
        assert!(!indexer.records_routing_decisions());

        let worker = WorkerWithDpRank::new(7, 0);
        let tokens = vec![1, 2, 3, 4];
        let block_hashes = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let mut tokens_with_hashes = TokensWithHashes::new(tokens, 4);

        indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer.find_matches_by_tier(block_hashes).await.unwrap();
        assert!(
            !matches.device.overlap_scores.scores.contains_key(&worker),
            "event-driven primary without side overlay should rely on KV events, not route-time writes"
        );
    }

    #[tokio::test]
    async fn side_only_worker_scored_but_not_used_as_lower_tier_anchor() {
        // Build an Indexer::Concurrent with a real side indexer so
        // `record_hashed_routing_decision` populates only the side path.
        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
        ));
        // PruneConfig is required to enable routing-decision recording on the
        // side indexer; without it the routing-decision path is a no-op.
        let side = Arc::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            1,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        ));
        let side_for_flush = side.clone();
        let indexer = Indexer::Concurrent {
            primary,
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: Some(super::SideIndexer::Concurrent(side)),
            primary_records_routing_decisions: false,
        };
        assert!(indexer.records_routing_decisions());

        let primary_worker = WorkerWithDpRank::new(10, 0);
        let side_only_worker = WorkerWithDpRank::new(20, 0);

        // Primary sees blocks [11, 12, 13] on Device for primary_worker;
        // extension block [14] on HostPinned for primary_worker.
        indexer
            .try_apply_event(store_event(
                10,
                0,
                1,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(
                10,
                0,
                2,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();
        // Crucially, also give side_only_worker a HostPinned extension at
        // block 14 anchored on the same prefix [11, 12, 13]. If the lower
        // tier were seeded from the side-merged device score, the host walk
        // would find this and credit a hit; with the reorder it should not.
        indexer
            .try_apply_event(store_event(
                20,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        // Side-only: route a decision so the side indexer learns
        // side_only_worker for the same device prefix. Primary never sees it.
        let block_hashes: Vec<LocalBlockHash> =
            [11, 12, 13].iter().copied().map(LocalBlockHash).collect();
        let sequence_hashes = compute_seq_hash_for_block(&block_hashes);
        indexer
            .record_hashed_routing_decision(side_only_worker, block_hashes.clone(), sequence_hashes)
            .await
            .unwrap();

        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
                LocalBlockHash(14),
            ])
            .await
            .unwrap();

        // Merge worked: both workers carry device scores.
        assert_eq!(
            matches
                .device
                .overlap_scores
                .scores
                .get(&primary_worker)
                .copied(),
            Some(3)
        );
        assert_eq!(
            matches
                .device
                .overlap_scores
                .scores
                .get(&side_only_worker)
                .copied(),
            Some(3),
            "side-only worker should appear in merged device scores"
        );

        // Reorder enforced: lower-tier was seeded from primary only.
        // primary_worker still extends into HostPinned via its own device
        // anchor. side_only_worker's HostPinned extension exists in the
        // host tier, but because the side score wasn't used as a device
        // anchor, the host walk does not start for it and its host hit is
        // not credited.
        let host = matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .expect("host-pinned tier should have been allocated");
        assert_eq!(host.hits.get(&primary_worker).copied(), Some(1));
        assert_eq!(
            host.hits.get(&side_only_worker).copied().unwrap_or(0),
            0,
            "side-only worker's host extension must not be credited \
             when lower-tier seeding is primary-only"
        );
        assert!(
            !host.next_continuations.contains_key(&side_only_worker),
            "side-only worker must not appear in lower-tier continuations"
        );
    }

    #[tokio::test]
    async fn borrowed_tiered_lookup_matches_owned_with_lower_tier_and_side_overlay() {
        let primary = Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            2,
            4,
        ));
        let side = Arc::new(ThreadPoolIndexer::new_with_pruning(
            ConcurrentRadixTreeCompressed::new(),
            1,
            4,
            PruneConfig {
                ttl: Duration::from_secs(60),
            },
        ));
        let side_for_flush = side.clone();
        let indexer = Indexer::Concurrent {
            primary,
            lower_tier: LowerTierIndexers::new(2, 4),
            approx: Some(super::SideIndexer::Concurrent(side)),
            primary_records_routing_decisions: false,
        };

        let primary_worker = WorkerWithDpRank::new(10, 0);
        let side_worker = WorkerWithDpRank::new(20, 0);
        indexer
            .try_apply_event(store_event(10, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(
                10,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await
            .unwrap();

        let side_hashes = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        indexer
            .record_routing_decision_hashes(
                side_worker,
                RoutingDecisionHashes {
                    local_hashes: side_hashes.clone(),
                    sequence_hashes: compute_seq_hash_for_block(&side_hashes),
                },
            )
            .await
            .unwrap();
        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let query = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        let borrowed = indexer.find_matches_by_tier_ref(&query).await.unwrap();
        let owned = indexer.find_matches_by_tier(query).await.unwrap();

        assert_eq!(
            borrowed.device.overlap_scores.scores,
            owned.device.overlap_scores.scores
        );
        assert_eq!(
            borrowed
                .lower_tier
                .get(&StorageTier::HostPinned)
                .map(|tier| &tier.hits),
            owned
                .lower_tier
                .get(&StorageTier::HostPinned)
                .map(|tier| &tier.hits)
        );
        assert_eq!(
            borrowed
                .device
                .overlap_scores
                .scores
                .get(&primary_worker)
                .copied(),
            Some(2)
        );
        assert_eq!(
            borrowed
                .device
                .overlap_scores
                .scores
                .get(&side_worker)
                .copied(),
            Some(3)
        );
    }

    #[tokio::test]
    async fn concurrent_tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores()
    {
        let indexer = make_test_concurrent_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .try_apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(21)])
            .await
            .unwrap();

        assert_eq!(
            matches.device.overlap_scores.scores.get(&device_worker),
            Some(&1)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&host_only_worker)
        );
        assert!(
            !matches
                .device
                .overlap_scores
                .scores
                .contains_key(&disk_only_worker)
        );

        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&host_only_worker)),
            Some(&1)
        );
        assert_eq!(
            matches
                .lower_tier
                .get(&StorageTier::Disk)
                .and_then(|tier| tier.hits.get(&disk_only_worker)),
            Some(&1)
        );
    }

    /// Regression test: when a worker has blocks in both device and lower-tier
    /// storage (e.g. same prefix stored on GPU and offloaded to host), the
    /// Concurrent indexer doesn't return last_matched_hashes. Without the fix,
    /// query_lower_tiers would re-query that worker from root in the lower tier,
    /// double-counting overlap blocks and producing cached_tokens > ISL.
    #[tokio::test]
    async fn concurrent_tiered_query_does_not_double_count_device_and_lower_tier_overlap() {
        let indexer = make_test_concurrent_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        // Device owns the prefix block; host-pinned extends it by one block.
        indexer
            .try_apply_event(store_event(7, 0, 1, &[], &[41], StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(store_event(7, 0, 2, &[41], &[42], StorageTier::HostPinned))
            .await
            .unwrap();
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![LocalBlockHash(41), LocalBlockHash(42)])
            .await
            .unwrap();

        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&1));

        let host_hits = matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .and_then(|tier| tier.hits.get(&worker).copied())
            .unwrap_or(0);
        assert_eq!(
            host_hits, 1,
            "lower-tier should extend the device prefix without double-counting it"
        );
    }
}
