// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    indexer::{
        KvIndexer, KvRouterError, LowerTierIndexers, MatchDetails, ThreadPoolIndexer,
        query_lower_tiers,
    },
    protocols::{LocalBlockHash, OverlapScores},
};

use super::{Indexer, SideIndexer, TieredMatchDetails, remote::RemoteIndexer};

#[derive(Clone, Copy)]
pub(super) enum PrimaryLookup<'a> {
    KvIndexer(&'a KvIndexer),
    Concurrent(&'a ThreadPoolIndexer<ConcurrentRadixTreeCompressed>),
    Remote(&'a RemoteIndexer),
    None,
}

#[derive(Clone, Copy)]
pub(super) struct LookupPipeline<'a> {
    primary: PrimaryLookup<'a>,
    lower_tier: Option<&'a LowerTierIndexers>,
    side: Option<&'a SideIndexer>,
}

impl Indexer {
    pub(super) fn lookup_pipeline(&self) -> LookupPipeline<'_> {
        match self {
            Self::KvIndexer {
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

    #[allow(dead_code)]
    pub(crate) async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.find_match_details(sequence)
            .await
            .map(|details| details.overlap_scores)
    }

    pub(crate) async fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline().find_match_details(sequence).await
    }

    pub(crate) async fn find_primary_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_match_details(sequence)
            .await
    }

    pub(crate) async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline().find_matches_by_tier(sequence).await
    }

    pub(crate) async fn find_primary_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_matches_by_tier(sequence)
            .await
    }
}

impl<'a> LookupPipeline<'a> {
    async fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = self.find_primary_match_details(sequence.clone()).await?;
        Ok(merge_side_or_warn(self.side, primary_details, sequence).await)
    }

    async fn find_primary_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.primary.find_match_details(sequence).await
    }

    async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
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
                let primary_device = self.find_primary_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &primary_device);
                let device = merge_side_or_warn(self.side, primary_device, sequence).await;

                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            PrimaryLookup::Remote(primary) => {
                let mut tiered = primary
                    .find_matches_by_tier(sequence.clone(), false)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Remote indexer tiered query failed");
                        KvRouterError::IndexerOffline
                    })?;
                tiered.device = merge_side_or_warn(self.side, tiered.device, sequence).await;
                Ok(tiered)
            }
            PrimaryLookup::None => Ok(TieredMatchDetails::default()),
        }
    }

    async fn find_primary_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self.primary {
            PrimaryLookup::KvIndexer(_) | PrimaryLookup::Concurrent(_) => {
                let Some(lower_tier) = self.lower_tier else {
                    return Ok(TieredMatchDetails::default());
                };
                let device = self.find_primary_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            PrimaryLookup::Remote(primary) => primary
                .find_matches_by_tier(sequence.clone(), false)
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
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = match self {
            Self::KvIndexer(primary) => primary.find_match_details(sequence.clone()).await?,
            Self::Concurrent(primary) => {
                primary.backend().find_match_details_impl(&sequence, false)
            }
            Self::Remote(primary) => {
                let tiered = primary
                    .find_matches_by_tier(sequence.clone(), true)
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
async fn merge_side_or_warn(
    side: Option<&SideIndexer>,
    primary: MatchDetails,
    sequence: Vec<LocalBlockHash>,
) -> MatchDetails {
    let Some(side) = side else {
        return primary;
    };
    match side.find_matches(sequence).await {
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
