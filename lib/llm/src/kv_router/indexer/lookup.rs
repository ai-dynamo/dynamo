// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    indexer::{
        KvIndexer, KvRouterError, LowerTierIndexers, LowerTierQueryOptions, MatchDetails,
        ThreadPoolIndexer, TieredMatchProvider, query_lower_tiers_with_options,
    },
    protocols::{LocalBlockHash, OverlapScores},
};
use dynamo_runtime::pipeline::async_trait;

use super::{Indexer, SideIndexer, TieredMatchDetails, remote::RemoteIndexer};

pub(super) enum HashInput<'a> {
    Borrowed(&'a [LocalBlockHash]),
    Owned(Vec<LocalBlockHash>),
}

impl<'a> HashInput<'a> {
    pub(super) fn as_slice(&self) -> &[LocalBlockHash] {
        match self {
            Self::Borrowed(sequence) => sequence,
            Self::Owned(sequence) => sequence,
        }
    }

    pub(super) fn clone_for_boundary(&self) -> Vec<LocalBlockHash> {
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

    #[cfg(test)]
    pub(crate) async fn find_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_match_details(HashInput::Owned(sequence))
            .await
    }

    pub(crate) async fn find_primary_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_match_details(HashInput::Owned(sequence))
            .await
    }

    pub(crate) async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_matches_by_tier_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    pub(crate) async fn find_matches_by_tier_with_options(
        &self,
        sequence: Vec<LocalBlockHash>,
        lower_tier_options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_matches_by_tier(HashInput::Owned(sequence), lower_tier_options)
            .await
    }

    pub(crate) async fn find_matches_by_tier_ref(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_matches_by_tier_ref_with_options(sequence, LowerTierQueryOptions::default())
            .await
    }

    pub(crate) async fn find_matches_by_tier_ref_with_options(
        &self,
        sequence: &[LocalBlockHash],
        lower_tier_options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_matches_by_tier(HashInput::Borrowed(sequence), lower_tier_options)
            .await
    }

    pub(crate) async fn find_primary_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_matches_by_tier(HashInput::Owned(sequence))
            .await
    }
}

#[async_trait]
impl TieredMatchProvider for Indexer {
    async fn find_tiered_matches(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.find_matches_by_tier_ref(sequence).await
    }
}

impl<'a> LookupPipeline<'a> {
    #[cfg(test)]
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
                        lower_tier_options.retain_router_hint_chain,
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
                if lower_tier_options.retain_router_hint_chain {
                    tracing::warn!(
                        "router_hint chain retention is not supported with remote primary indexer; proceeding without router hints"
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
        retain_router_hint_chain: bool,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = match self {
            Self::KvIndexer(primary) => {
                primary
                    .find_match_details_with_options(
                        sequence.clone_for_boundary(),
                        retain_router_hint_chain,
                    )
                    .await?
            }
            Self::Concurrent(primary) => primary.backend().find_match_details_impl_with_options(
                sequence.as_slice(),
                false,
                retain_router_hint_chain,
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
async fn merge_side_or_warn(
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
