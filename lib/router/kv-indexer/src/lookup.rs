// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Match lookups over an [`Indexer`]: primary device tier, lower tiers seeded
//! from confirmed primary matches, and the predict-on-route side overlay.

use crate::prelude::{LocalBlockHash, OverlapScores};
use crate::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};
use crate::{
    KvCacheGroupResidency, KvIndexer, KvRouterError, LowerTierIndexers, LowerTierQueryOptions,
    MatchDetails, TieredMatchDetails, query_lower_tiers_with_hybrid,
    query_lower_tiers_with_options,
};

use super::backend::{Indexer, RemotePrimary, SideIndexer};
use crate::forward_prefix::ForwardDeviceEvaluator;
use crate::hybrid_prefix::{HybridPrefixEvaluator, cap_side_overlay};

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
    recurrent_layouts: Option<&'a KvCacheGroupResidency>,
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
                recurrent_layouts: Some(lower_tier.group_residency()),
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
                recurrent_layouts: Some(lower_tier.group_residency()),
            },
            Self::Remote {
                primary,
                approx,
                recurrent_layouts,
                ..
            } => LookupPipeline {
                primary: PrimaryLookup::Remote(primary.as_ref()),
                lower_tier: None,
                side: approx.as_ref(),
                recurrent_layouts: Some(recurrent_layouts),
            },
            Self::None => LookupPipeline {
                primary: PrimaryLookup::None,
                lower_tier: None,
                side: None,
                recurrent_layouts: None,
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

    pub async fn find_primary_match_details_ref(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .find_primary_match_details(HashInput::Borrowed(sequence))
            .await
    }

    /// Device match details plus per-tier hits, with copy-source chain retention
    /// disabled. Use [`Indexer::find_tiered_matches_with_options`] to request
    /// retention through the capability-gated variant.
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

impl Indexer {
    pub async fn find_primary_match_details_with_limit(
        &self,
        sequence: Vec<LocalBlockHash>,
        max_blocks: Option<usize>,
    ) -> Result<MatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .primary_device(&HashInput::Owned(sequence), false, max_blocks, false)
            .await
    }
    pub async fn find_primary_matches_by_tier_with_limit(
        &self,
        sequence: Vec<LocalBlockHash>,
        max_blocks: Option<usize>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        self.lookup_pipeline()
            .local_tiered(
                &HashInput::Owned(sequence),
                LowerTierQueryOptions {
                    retain_kv_transfer_chain: false,
                    max_reusable_blocks: max_blocks,
                },
                false,
            )
            .await
    }
}

impl<'a> LookupPipeline<'a> {
    fn group_residency(&self) -> Option<&'a KvCacheGroupResidency> {
        self.recurrent_layouts
            .filter(|residency| residency.is_active())
    }

    async fn merge_side(&self, primary: MatchDetails, sequence: HashInput<'_>) -> MatchDetails {
        if self.side.is_none() {
            return primary;
        }
        let needs_cap = self.group_residency().is_some() || !primary.hybrid_resume_caps.is_empty();
        if !needs_cap {
            return merge_side_or_warn(self.side, primary, sequence).await;
        }
        let confirmed = primary.overlap_scores.scores.clone();
        let mut merged = merge_side_or_warn(self.side, primary, sequence).await;
        cap_side_overlay(self.group_residency(), &mut merged, &confirmed);
        merged
    }

    async fn find_match_details(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        if self.side.is_none() && self.group_residency().is_none() {
            return self.primary.find_match_details_owned(sequence).await;
        }
        self.primary_device(&sequence, false, None, true).await
    }

    async fn primary_device(
        &self,
        sequence: &HashInput<'_>,
        retain: bool,
        max_blocks: Option<usize>,
        with_side: bool,
    ) -> Result<MatchDetails, KvRouterError> {
        let residency = self.group_residency().filter(|_| self.lower_tier.is_some());
        let mut details = if let Some(residency) = residency {
            if (!with_side || self.side.is_none())
                && let Some(details) = self
                    .primary
                    .find_forward_device(sequence, retain, max_blocks, residency)
            {
                return Ok(details);
            }
            let evaluator = HybridPrefixEvaluator::new(residency, max_blocks);
            let (mut raw, mut evaluator) = self
                .primary
                .find_observed(sequence, retain, evaluator)
                .await?;
            evaluator.prepare_device(&raw);
            if with_side {
                raw = merge_side_or_warn(self.side, raw, HashInput::Borrowed(sequence.as_slice()))
                    .await;
            }
            evaluator.finish(&mut raw, &mut Default::default(), &Default::default());
            raw
        } else {
            let raw = self
                .primary
                .find_match_details_retained_with_options(sequence, retain, max_blocks)
                .await?;
            if with_side {
                self.merge_side(raw, HashInput::Borrowed(sequence.as_slice()))
                    .await
            } else {
                raw
            }
        };
        if !retain {
            details.kv_transfer_candidates = None;
        }
        Ok(details)
    }

    async fn find_primary_match_details(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        if self.group_residency().is_none() {
            return self.primary.find_match_details_owned(sequence).await;
        }
        self.primary_device(&sequence, false, None, false).await
    }

    async fn local_tiered(
        &self,
        sequence: &HashInput<'_>,
        options: LowerTierQueryOptions,
        with_side: bool,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        let Some(lower_tier) = self.lower_tier else {
            return Ok(TieredMatchDetails::default());
        };
        let (mut device, lower) = if let Some(residency) = self.group_residency() {
            if lower_tier.is_empty()
                && (!with_side || self.side.is_none())
                && let Some(device) = self.primary.find_forward_device(
                    sequence,
                    options.retain_kv_transfer_chain,
                    options.max_reusable_blocks,
                    residency,
                )
            {
                return Ok(TieredMatchDetails {
                    device,
                    lower_tier: Default::default(),
                });
            }
            let evaluator = HybridPrefixEvaluator::new(residency, options.max_reusable_blocks);
            let (mut raw, mut evaluator) = self
                .primary
                .find_observed(sequence, options.retain_kv_transfer_chain, evaluator)
                .await?;
            let (mut lower, snapshot) = if lower_tier.is_empty() {
                evaluator.prepare_device(&raw);
                (Default::default(), None)
            } else {
                let anchor = evaluator.query_anchor(&raw);
                let snapshot = lower_tier.routing_snapshot();
                let lower = query_lower_tiers_with_hybrid(
                    lower_tier,
                    sequence.as_slice(),
                    &anchor,
                    options,
                    snapshot.clone(),
                    &mut evaluator,
                );
                (lower, Some(snapshot))
            };
            if with_side {
                raw = merge_side_or_warn(self.side, raw, HashInput::Borrowed(sequence.as_slice()))
                    .await;
            }
            evaluator.finish(
                &mut raw,
                &mut lower,
                snapshot.as_deref().unwrap_or(&Default::default()),
            );
            (raw, lower)
        } else {
            let mut raw = self
                .primary
                .find_match_details_retained_with_options(
                    sequence,
                    options.retain_kv_transfer_chain,
                    options.max_reusable_blocks,
                )
                .await?;
            let lower =
                query_lower_tiers_with_options(lower_tier, sequence.as_slice(), &raw, options);
            if with_side {
                raw = self
                    .merge_side(raw, HashInput::Borrowed(sequence.as_slice()))
                    .await;
            }
            (raw, lower)
        };
        if !options.retain_kv_transfer_chain {
            device.kv_transfer_candidates = None;
        }
        Ok(TieredMatchDetails {
            device,
            lower_tier: lower,
        })
    }

    async fn find_matches_by_tier(
        &self,
        sequence: HashInput<'_>,
        options: LowerTierQueryOptions,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        let mut tiered = match self.primary {
            PrimaryLookup::KvIndexer(_) | PrimaryLookup::Concurrent(_) => {
                return self.local_tiered(&sequence, options, true).await;
            }
            PrimaryLookup::Remote(primary) => primary
                .find_matches_by_tier_with_limit(
                    sequence.clone_for_boundary(),
                    false,
                    options.max_reusable_blocks,
                )
                .await
                .map_err(|error| {
                    tracing::warn!(%error, "Remote indexer tiered query failed");
                    KvRouterError::IndexerOffline
                })?,
            PrimaryLookup::None => return Ok(TieredMatchDetails::default()),
        };
        tiered.device = self.merge_side(tiered.device, sequence).await;
        Ok(tiered)
    }

    async fn find_primary_matches_by_tier(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self.primary {
            PrimaryLookup::KvIndexer(_) | PrimaryLookup::Concurrent(_) => {
                self.local_tiered(&sequence, LowerTierQueryOptions::default(), false)
                    .await
            }
            PrimaryLookup::Remote(primary) => primary
                .find_matches_by_tier(sequence.into_owned_at_boundary(), false)
                .await
                .map_err(|error| {
                    tracing::warn!(%error, "Remote indexer tiered query failed");
                    KvRouterError::IndexerOffline
                }),
            PrimaryLookup::None => Ok(TieredMatchDetails::default()),
        }
    }
}

impl<'a> PrimaryLookup<'a> {
    fn find_forward_device(
        &self,
        sequence: &HashInput<'_>,
        retain: bool,
        max_blocks: Option<usize>,
        residency: &KvCacheGroupResidency,
    ) -> Option<MatchDetails> {
        let Self::Concurrent(primary) = self else {
            return None;
        };
        let mut evaluator = ForwardDeviceEvaluator::new(residency, max_blocks);
        let mut details = primary.backend().find_match_details_impl_with_observer(
            sequence.as_slice(),
            false,
            retain,
            &mut evaluator,
        );
        evaluator.finish(&mut details);
        Some(details)
    }

    async fn find_observed(
        &self,
        sequence: &HashInput<'_>,
        retain: bool,
        mut evaluator: HybridPrefixEvaluator,
    ) -> Result<(MatchDetails, HybridPrefixEvaluator), KvRouterError> {
        match self {
            Self::KvIndexer(primary) => {
                primary
                    .find_match_details_with_observer(
                        sequence.clone_for_boundary(),
                        retain,
                        evaluator,
                    )
                    .await
            }
            Self::Concurrent(primary) => {
                let details = primary.backend().find_match_details_impl_with_observer(
                    sequence.as_slice(),
                    false,
                    retain,
                    &mut evaluator,
                );
                Ok((details, evaluator))
            }
            _ => unreachable!("only local primaries collect checkpoint evidence"),
        }
    }

    async fn find_match_details_owned(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<MatchDetails, KvRouterError> {
        match self {
            Self::KvIndexer(primary) => {
                primary
                    .find_match_details(sequence.into_owned_at_boundary())
                    .await
            }
            Self::Concurrent(primary) => Ok(primary
                .backend()
                .find_match_details_impl(sequence.as_slice(), false)),
            Self::Remote(primary) => primary
                .find_matches_by_tier(sequence.into_owned_at_boundary(), true)
                .await
                .map(|tiered| tiered.device)
                .map_err(|_| KvRouterError::IndexerOffline),
            Self::None => Ok(MatchDetails::default()),
        }
    }

    async fn find_match_details_retained_with_options(
        &self,
        sequence: &HashInput<'_>,
        retain_kv_transfer_chain: bool,
        max_blocks: Option<usize>,
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
                    .find_matches_by_tier_with_limit(
                        sequence.clone_for_boundary(),
                        true,
                        max_blocks,
                    )
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
/// `query_lower_tiers_with_options`, which assumes the lockstep invariant. The local
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
    use super::{ForwardDeviceEvaluator, HybridPrefixEvaluator};
    use std::sync::Arc;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use crate::backend::Indexer;
    use crate::backend::test_util::store_event;
    use crate::prelude::{
        BlockHashOptions, LocalBlockHash, StorageTier, TokensWithHashes, WorkerWithDpRank,
        compute_block_hash_for_seq, compute_seq_hash_for_block,
    };
    use crate::testing::recurrent_store as checkpoint;
    use crate::{
        ConcurrentRadixTreeCompressed, KvIndexer, KvIndexerInterface, KvIndexerMetrics,
        RoutingDecisionHashes, ThreadPoolIndexer, pruning::PruneConfig,
    };
    use crate::{LowerTierIndexers, LowerTierQueryOptions};

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
            session_updates: None,
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
            session_updates: None,
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
            session_updates: None,
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

    #[tokio::test]
    async fn forward_presence_matches_general_evaluator_and_observes_cleared_state() {
        use crate::HybridResumeCaps;

        for missing in [false, true] {
            let indexer = make_test_concurrent_indexer();
            let sequence: Vec<_> = [11, 12, 13].into_iter().map(LocalBlockHash).collect();
            let chain = compute_seq_hash_for_block(&sequence);
            let workers = [WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)];
            // Declare every layout before storing the common checkpoint.
            for worker in workers {
                indexer.declare_recurrent_layout(worker, &[0, 1], 1);
            }
            for worker in workers {
                indexer
                    .try_apply_event(store_event(
                        worker.worker_id,
                        0,
                        1,
                        &[],
                        &[11, 12, 13],
                        StorageTier::Device,
                    ))
                    .await
                    .unwrap();
                for group in [0, 1] {
                    indexer
                        .try_apply_event(checkpoint(
                            worker,
                            StorageTier::Device,
                            if missing && worker == workers[1] {
                                chain[1]
                            } else {
                                chain[2]
                            },
                            group,
                            2,
                        ))
                        .await
                        .unwrap();
                }
            }
            flush_indexer(&indexer).await;
            let Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } = &indexer
            else {
                unreachable!()
            };
            let mut evaluator = HybridPrefixEvaluator::new(lower_tier.group_residency(), None);
            let mut expected = primary.backend().find_match_details_impl_with_observer(
                &sequence,
                false,
                true,
                &mut evaluator,
            );
            evaluator.prepare_device(&expected);
            evaluator.finish(&mut expected, &mut Default::default(), &Default::default());
            let actual = indexer
                .find_matches_by_tier_with_options(
                    sequence.clone(),
                    LowerTierQueryOptions {
                        retain_kv_transfer_chain: true,
                        max_reusable_blocks: None,
                    },
                )
                .await
                .unwrap()
                .device;
            assert_eq!(actual.overlap_scores.scores, expected.overlap_scores.scores);
            assert_eq!(actual.last_matched_hashes, expected.last_matched_hashes);
            assert_eq!(actual.hybrid_resume_caps, expected.hybrid_resume_caps);
            assert_eq!(
                actual.kv_transfer_candidates,
                expected.kv_transfer_candidates
            );
            if !missing {
                assert!(matches!(
                    actual.hybrid_resume_caps,
                    HybridResumeCaps::UniformConfirmed { .. }
                ));
                lower_tier.group_residency().clear_worker(workers[0]);
                let mut observer = ForwardDeviceEvaluator::new(lower_tier.group_residency(), None);
                let mut after_clear = primary.backend().find_match_details_impl_with_observer(
                    &sequence,
                    false,
                    false,
                    &mut observer,
                );
                observer.finish(&mut after_clear);
                assert!(!after_clear.overlap_scores.scores.contains_key(&workers[0]));
                assert_eq!(after_clear.hybrid_resume_caps.get(&workers[0]), Some(&0));
            } else {
                assert!(matches!(
                    actual.hybrid_resume_caps,
                    HybridResumeCaps::SparseConfirmed(_)
                ));
            }
        }
    }

    #[tokio::test]
    async fn bounded_forward_presence_matches_general_caps_and_keeps_dense_coverage() {
        use crate::HybridResumeCaps;

        for with_dense_worker in [false, true] {
            let indexer = make_test_concurrent_indexer();
            let sequence: Vec<_> = [11, 12, 13].into_iter().map(LocalBlockHash).collect();
            let chain = compute_seq_hash_for_block(&sequence);
            for id in [1, 2] {
                indexer.declare_recurrent_layout(WorkerWithDpRank::new(id, 0), &[0], 1);
            }
            for id in 1..=if with_dense_worker { 3 } else { 2 } {
                indexer
                    .try_apply_event(store_event(
                        id,
                        0,
                        1,
                        &[],
                        &[11, 12, 13],
                        StorageTier::Device,
                    ))
                    .await
                    .unwrap();
                if id != 3 {
                    for &hash in &chain {
                        indexer
                            .try_apply_event(checkpoint(
                                WorkerWithDpRank::new(id, 0),
                                StorageTier::Device,
                                hash,
                                0,
                                1,
                            ))
                            .await
                            .unwrap();
                    }
                }
            }
            flush_indexer(&indexer).await;
            let Indexer::Concurrent {
                primary,
                lower_tier,
                ..
            } = &indexer
            else {
                unreachable!()
            };
            for limit in [Some(0), Some(1), Some(2), Some(3), Some(8), None] {
                let mut evaluator = HybridPrefixEvaluator::new(lower_tier.group_residency(), limit);
                let mut expected = primary.backend().find_match_details_impl_with_observer(
                    &sequence,
                    false,
                    true,
                    &mut evaluator,
                );
                evaluator.prepare_device(&expected);
                evaluator.finish(&mut expected, &mut Default::default(), &Default::default());
                let actual = indexer
                    .find_matches_by_tier_with_options(
                        sequence.clone(),
                        LowerTierQueryOptions {
                            retain_kv_transfer_chain: true,
                            max_reusable_blocks: limit,
                        },
                    )
                    .await
                    .unwrap()
                    .device;
                assert_eq!(
                    actual.overlap_scores.scores, expected.overlap_scores.scores,
                    "dense={with_dense_worker}, limit={limit:?}"
                );
                assert_eq!(actual.last_matched_hashes, expected.last_matched_hashes);
                assert_eq!(actual.hybrid_resume_caps, expected.hybrid_resume_caps);
                assert_eq!(
                    actual.kv_transfer_candidates,
                    expected.kv_transfer_candidates
                );
                if with_dense_worker {
                    assert_eq!(
                        actual
                            .overlap_scores
                            .scores
                            .get(&WorkerWithDpRank::new(3, 0)),
                        Some(&3)
                    );
                } else if limit != Some(0) {
                    assert!(matches!(
                        actual.hybrid_resume_caps,
                        HybridResumeCaps::UniformConfirmed { .. }
                    ));
                }
            }
        }
    }

    #[tokio::test]
    async fn device_only_hints_match_with_and_without_an_allocated_empty_lower_tier() {
        let worker = WorkerWithDpRank::new(7, 0);
        let sequence: Vec<_> = [11, 12, 13].into_iter().map(LocalBlockHash).collect();
        let chain = compute_seq_hash_for_block(&sequence);
        for concurrent in [false, true] {
            for allocated in [false, true] {
                let indexer = if concurrent {
                    make_test_concurrent_indexer()
                } else {
                    make_test_indexer()
                };
                indexer.declare_recurrent_groups(worker, &[0]);
                indexer
                    .try_apply_event(store_event(
                        7,
                        0,
                        1,
                        &[],
                        &[11, 12, 13],
                        StorageTier::Device,
                    ))
                    .await
                    .unwrap();
                indexer
                    .try_apply_event(checkpoint(worker, StorageTier::Device, chain[1], 0, 1))
                    .await
                    .unwrap();
                if allocated {
                    match &indexer {
                        Indexer::Single { lower_tier, .. }
                        | Indexer::Concurrent { lower_tier, .. } => {
                            lower_tier.get_or_create(StorageTier::HostPinned);
                        }
                        _ => unreachable!(),
                    }
                }
                flush_indexer(&indexer).await;

                let result = indexer
                    .find_matches_by_tier_with_options(
                        sequence.clone(),
                        LowerTierQueryOptions {
                            retain_kv_transfer_chain: true,
                            max_reusable_blocks: None,
                        },
                    )
                    .await
                    .unwrap();
                assert_eq!(result.device.overlap_scores.scores.get(&worker), Some(&2));
                assert_eq!(result.device.hybrid_resume_caps.get(&worker), Some(&2));
                assert_eq!(
                    result.device.last_matched_hashes.get(&worker),
                    Some(&crate::prelude::ExternalSequenceBlockHash(chain[1]))
                );
                assert!(
                    result
                        .lower_tier
                        .values()
                        .all(|details| details.hits.values().all(|hits| *hits == 0))
                );
                let hints = result.kv_transfer_candidates().unwrap();
                assert_eq!(
                    hints.block_hashes,
                    chain[..2]
                        .iter()
                        .copied()
                        .map(crate::prelude::ExternalSequenceBlockHash)
                        .collect::<Vec<_>>()
                );
                assert_eq!(hints.owner_prefix_blocks, vec![(worker.into(), 2)]);
            }
        }
    }

    #[tokio::test]
    async fn hybrid_cumulative_cut_uses_one_boundary_for_host_and_disk() {
        for disk_fills_gap in [false, true] {
            let indexer = make_test_indexer();
            let worker = WorkerWithDpRank::new(7, 0);
            let sequence: Vec<_> = [11, 12, 13, 14, 15]
                .into_iter()
                .map(LocalBlockHash)
                .collect();
            let chain = compute_seq_hash_for_block(&sequence);
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
                    &[13, 14],
                    StorageTier::HostPinned,
                ))
                .await
                .unwrap();
            indexer
                .try_apply_event(store_event(
                    7,
                    0,
                    3,
                    &[11, 12, 13, 14],
                    &[15],
                    StorageTier::Disk,
                ))
                .await
                .unwrap();
            for (tier, pos) in [
                (StorageTier::Device, 1),
                (StorageTier::HostPinned, 2),
                (StorageTier::Disk, 4),
            ] {
                indexer
                    .try_apply_event(checkpoint(
                        worker,
                        tier,
                        if tier == StorageTier::Disk && !disk_fills_gap {
                            0xdead
                        } else {
                            chain[pos]
                        },
                        0,
                        1,
                    ))
                    .await
                    .unwrap();
            }
            flush_indexer(&indexer).await;
            let result = indexer
                .find_tiered_matches_with_options(
                    sequence,
                    LowerTierQueryOptions {
                        retain_kv_transfer_chain: true,
                        max_reusable_blocks: None,
                    },
                )
                .await
                .unwrap();
            let candidates = result
                .kv_transfer_candidates()
                .expect("host hints retained");
            assert!(
                candidates
                    .owner_prefix_blocks
                    .iter()
                    .all(|(_, blocks)| *blocks <= 3),
                "host-only hints need a host checkpoint"
            );
            assert!(
                candidates
                    .source_intervals
                    .iter()
                    .all(|interval| interval.end <= 3)
            );
            assert_eq!(result.device.overlap_scores.scores[&worker], 2);
            assert_eq!(
                result.lower_tier[&StorageTier::HostPinned].hits[&worker],
                if disk_fills_gap { 2 } else { 1 }
            );
            assert_eq!(
                result.lower_tier[&StorageTier::Disk]
                    .hits
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
                if disk_fills_gap { 1 } else { 0 }
            );
        }
    }

    #[tokio::test]
    async fn cpu_checkpoint_does_not_grant_device_credit_and_prompt_limit_survives_wrappers() {
        let indexer = make_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);
        let sequence: Vec<_> = [11, 12, 13].into_iter().map(LocalBlockHash).collect();
        let chain = compute_seq_hash_for_block(&sequence);
        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            indexer
                .try_apply_event(store_event(7, 0, 1, &[], &[11, 12, 13], tier))
                .await
                .unwrap();
        }
        indexer.declare_recurrent_groups(worker, &[0, 1, 2]);
        for group in [0, 1, 2] {
            for hash in [chain[1], chain[2]] {
                indexer
                    .try_apply_event(checkpoint(worker, StorageTier::HostPinned, hash, group, 3))
                    .await
                    .unwrap();
            }
        }
        flush_indexer(&indexer).await;
        let uncapped = indexer.find_tiered_matches(sequence.clone()).await.unwrap();
        assert!(!uncapped.device.overlap_scores.scores.contains_key(&worker));
        assert_eq!(
            uncapped.lower_tier[&StorageTier::HostPinned].hits[&worker],
            3
        );
        for retain in [false, true] {
            let result = indexer
                .find_tiered_matches_with_options(
                    sequence.clone(),
                    LowerTierQueryOptions {
                        retain_kv_transfer_chain: retain,
                        max_reusable_blocks: Some(2),
                    },
                )
                .await
                .unwrap();
            assert!(!result.device.overlap_scores.scores.contains_key(&worker));
            assert_eq!(result.lower_tier[&StorageTier::HostPinned].hits[&worker], 2);
            if let Some(candidates) = result.kv_transfer_candidates() {
                assert!(
                    candidates
                        .owner_prefix_blocks
                        .iter()
                        .all(|(_, blocks)| *blocks <= 2)
                );
                assert!(
                    candidates
                        .source_intervals
                        .iter()
                        .all(|interval| interval.end <= 2)
                );
            }
        }
        indexer.reset_worker_dp_rank_and_wait(7, 0).await.unwrap();
        indexer
            .try_apply_event(store_event(
                7,
                0,
                9,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await
            .unwrap();
        flush_indexer(&indexer).await;
        assert!(
            !indexer
                .find_tiered_matches(sequence)
                .await
                .unwrap()
                .device
                .overlap_scores
                .scores
                .contains_key(&worker),
            "reset must preserve an empty hybrid layout"
        );
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

    /// Predicted attention cannot lift confirmed recurrent credit.
    #[tokio::test]
    async fn side_merge_does_not_lift_hybrid_worker_above_its_checkpoint() {
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
            session_updates: None,
        };
        let hybrid = WorkerWithDpRank::new(10, 0);
        let local = [11_u64, 12, 13];
        indexer
            .try_apply_event(store_event(10, 0, 1, &[], &local, StorageTier::Device))
            .await
            .unwrap();
        let block_hashes: Vec<LocalBlockHash> = local.iter().copied().map(LocalBlockHash).collect();
        let chain = compute_seq_hash_for_block(&block_hashes);
        indexer
            .try_apply_event(checkpoint(hybrid, StorageTier::Device, chain[0], 1, 1))
            .await
            .unwrap();
        indexer
            .record_hashed_routing_decision(hybrid, block_hashes.clone(), chain.clone())
            .await
            .unwrap();
        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let tiered = indexer
            .find_matches_by_tier(block_hashes.clone())
            .await
            .unwrap();
        assert_eq!(
            tiered.device.overlap_scores.scores.get(&hybrid).copied(),
            Some(1)
        );
        assert!(tiered.device.kv_transfer_candidates.is_none());
        let details = indexer
            .find_match_details(block_hashes.clone())
            .await
            .unwrap();
        assert_eq!(details.overlap_scores.scores.get(&hybrid).copied(), Some(1));
        let scores = indexer.find_matches(block_hashes).await.unwrap();
        assert_eq!(scores.scores.get(&hybrid).copied(), Some(1));
    }

    /// Off-prefix checkpoints cannot revive a predicted match.
    #[tokio::test]
    async fn side_merge_does_not_revive_hybrid_worker_without_state_on_prefix() {
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
            session_updates: None,
        };
        let hybrid = WorkerWithDpRank::new(10, 0);
        let local = [11_u64, 12, 13];
        indexer
            .try_apply_event(store_event(10, 0, 1, &[], &local, StorageTier::Device))
            .await
            .unwrap();
        indexer
            .try_apply_event(checkpoint(hybrid, StorageTier::Device, 0xdead_beef, 1, 1))
            .await
            .unwrap();
        let block_hashes: Vec<LocalBlockHash> = local.iter().copied().map(LocalBlockHash).collect();
        let chain = compute_seq_hash_for_block(&block_hashes);
        indexer
            .record_hashed_routing_decision(hybrid, block_hashes.clone(), chain)
            .await
            .unwrap();
        flush_indexer(&indexer).await;
        side_for_flush.flush().await;

        let tiered = indexer
            .find_matches_by_tier(block_hashes.clone())
            .await
            .unwrap();
        assert!(
            !tiered.device.overlap_scores.scores.contains_key(&hybrid),
            "tiered: {:?}",
            tiered.device.overlap_scores.scores
        );
        let details = indexer
            .find_match_details(block_hashes.clone())
            .await
            .unwrap();
        assert!(!details.overlap_scores.scores.contains_key(&hybrid));
        let scores = indexer.find_matches(block_hashes).await.unwrap();
        assert!(!scores.scores.contains_key(&hybrid));
    }

    /// Every lower-tier extension needs a usable checkpoint.
    #[tokio::test]
    async fn lower_tier_cut_covers_every_tier() {
        for tier in [
            StorageTier::HostPinned,
            StorageTier::Disk,
            StorageTier::External,
        ] {
            let indexer = make_test_indexer();
            let worker = WorkerWithDpRank::new(7, 0);
            indexer
                .try_apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
                .await
                .unwrap();
            indexer
                .try_apply_event(store_event(7, 0, 2, &[11, 12], &[13, 14], tier))
                .await
                .unwrap();
            let sequence: Vec<LocalBlockHash> = [11_u64, 12, 13, 14]
                .into_iter()
                .map(LocalBlockHash)
                .collect();
            let chain = compute_seq_hash_for_block(&sequence);
            indexer
                .try_apply_event(checkpoint(worker, StorageTier::Device, chain[1], 1, 1))
                .await
                .unwrap();
            flush_indexer(&indexer).await;

            let matches = indexer.find_matches_by_tier(sequence).await.unwrap();
            assert_eq!(
                matches.device.overlap_scores.scores.get(&worker),
                Some(&2),
                "{tier:?}: device match ends at the state"
            );
            let tier_details = matches.lower_tier.get(&tier).expect("tier queried");
            assert_eq!(
                tier_details.hits.get(&worker).copied().unwrap_or(0),
                0,
                "{tier:?}: no state past block 2, so no tier hit is credited"
            );
            assert_eq!(
                tier_details
                    .next_continuations
                    .get(&worker)
                    .map(|c| c.start_pos),
                Some(2),
                "{tier:?}: continuation cut back to the state boundary"
            );
            assert!(
                tier_details.kv_transfer_extensions.is_none(),
                "{tier:?}: extensions retained for the cut only"
            );
        }
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
            session_updates: None,
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
            session_updates: None,
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
        let borrowed = indexer
            .find_matches_by_tier_ref_with_options(&query, LowerTierQueryOptions::default())
            .await
            .unwrap();
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

    #[tokio::test]
    async fn remote_wire_caps_survive_local_predictions_without_a_local_hybrid_layout() {
        use super::super::backend::{RemotePrimary, SideIndexer};
        use crate::{
            KvCacheGroupResidency, LowerTierMatchDetails, MatchDetails, TieredMatchDetails,
            WireTieredMatchDetails,
        };

        type QueryLimit = (bool, Option<usize>);

        struct WireRemote {
            wire: WireTieredMatchDetails,
            limits: Arc<std::sync::Mutex<Vec<QueryLimit>>>,
        }

        #[async_trait::async_trait]
        impl RemotePrimary for WireRemote {
            async fn find_matches_by_tier(
                &self,
                _hashes: Vec<LocalBlockHash>,
                device_only: bool,
            ) -> anyhow::Result<TieredMatchDetails> {
                // Exercise the transport representation rather than passing the
                // native caps directly from the server to the caller.
                let encoded = serde_json::to_vec(&self.wire)?;
                let wire: WireTieredMatchDetails = serde_json::from_slice(&encoded)?;
                let mut details: TieredMatchDetails = wire.into();
                if device_only {
                    details.lower_tier.clear();
                }
                Ok(details)
            }

            async fn find_matches_by_tier_with_limit(
                &self,
                hashes: Vec<LocalBlockHash>,
                device_only: bool,
                max_reusable_blocks: Option<usize>,
            ) -> anyhow::Result<TieredMatchDetails> {
                self.limits
                    .lock()
                    .unwrap()
                    .push((device_only, max_reusable_blocks));
                self.find_matches_by_tier(hashes, device_only).await
            }

            async fn record_routing_decision(
                &self,
                _worker: WorkerWithDpRank,
                _hashes: RoutingDecisionHashes,
            ) -> anyhow::Result<()> {
                anyhow::bail!("predictions belong in the caller's side indexer")
            }

            fn use_kv_events(&self) -> bool {
                true
            }
        }

        let capped = WorkerWithDpRank::new(7, 0);
        let zero = WorkerWithDpRank::new(8, 0);
        let dense = WorkerWithDpRank::new(9, 0);
        let mut device = MatchDetails::default();
        device.overlap_scores.scores.insert(capped, 1);
        device.hybrid_resume_caps.extend([(capped, 1), (zero, 0)]);
        let mut host = LowerTierMatchDetails::default();
        host.hits.extend([(capped, 2), (zero, 3)]);
        let served = TieredMatchDetails {
            device,
            lower_tier: [(StorageTier::HostPinned, host)].into_iter().collect(),
        };
        let remote_wire = WireTieredMatchDetails::from(&served);
        let local_layouts = Arc::new(KvCacheGroupResidency::new());
        assert!(!local_layouts.is_active());
        let limits = Arc::new(std::sync::Mutex::new(Vec::new()));
        let indexer = Indexer::Remote {
            primary: Arc::new(WireRemote {
                wire: remote_wire,
                limits: limits.clone(),
            }),
            approx: Some(SideIndexer::new(
                Duration::from_secs(60),
                4,
                1,
                Arc::new(KvIndexerMetrics::new_unregistered()),
                CancellationToken::new(),
            )),
            primary_records_routing_decisions: false,
            recurrent_layouts: local_layouts,
        };
        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12), LocalBlockHash(13)];
        for worker in [capped, zero, dense] {
            indexer
                .record_routing_decision_hashes(
                    worker,
                    RoutingDecisionHashes::from_local_hashes(sequence.clone()),
                )
                .await
                .unwrap();
        }
        if let Indexer::Remote {
            approx: Some(SideIndexer::KvIndexer(side)),
            ..
        } = &indexer
        {
            side.flush().await;
        }

        let result = indexer
            .find_matches_by_tier(sequence.clone())
            .await
            .unwrap();
        assert_eq!(result.device.overlap_scores.scores.get(&capped), Some(&1));
        assert!(!result.device.overlap_scores.scores.contains_key(&zero));
        assert_eq!(result.device.overlap_scores.scores.get(&dense), Some(&3));
        assert_eq!(result.device.hybrid_resume_caps.get(&zero), Some(&0));
        assert_eq!(
            result.lower_tier[&StorageTier::HostPinned]
                .hits
                .get(&capped),
            Some(&2)
        );
        assert_eq!(
            result.lower_tier[&StorageTier::HostPinned].hits.get(&zero),
            Some(&3)
        );

        let device_only = indexer.find_match_details(sequence.clone()).await.unwrap();
        assert_eq!(device_only.overlap_scores.scores.get(&capped), Some(&1));
        assert!(!device_only.overlap_scores.scores.contains_key(&zero));
        assert_eq!(device_only.overlap_scores.scores.get(&dense), Some(&3));

        let limited_device = indexer
            .find_primary_match_details_with_limit(sequence.clone(), Some(2))
            .await
            .unwrap();
        assert_eq!(limited_device.overlap_scores.scores.get(&capped), Some(&1));
        let limited_tiers = indexer
            .find_matches_by_tier_with_options(
                sequence,
                LowerTierQueryOptions {
                    max_reusable_blocks: Some(3),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            limited_tiers.lower_tier[&StorageTier::HostPinned]
                .hits
                .get(&capped),
            Some(&2)
        );
        assert_eq!(
            *limits.lock().unwrap(),
            [
                (false, None),
                (true, None),
                (true, Some(2)),
                (false, Some(3))
            ]
        );
    }

    #[test]
    fn old_tiered_wire_response_defaults_to_no_hybrid_caps() {
        use crate::{TieredMatchDetails, WireTieredMatchDetails};

        let mut old_wire = serde_json::to_value(WireTieredMatchDetails::default()).unwrap();
        old_wire
            .as_object_mut()
            .unwrap()
            .remove("hybrid_resume_caps");
        let decoded: WireTieredMatchDetails = serde_json::from_value(old_wire).unwrap();
        assert!(decoded.hybrid_resume_caps.is_empty());
        let native: TieredMatchDetails = decoded.into();
        assert!(native.device.hybrid_resume_caps.is_empty());
    }

    /// Regression test: when a worker has blocks in both device and lower-tier
    /// storage (e.g. same prefix stored on GPU and offloaded to host), the
    /// Concurrent indexer doesn't return last_matched_hashes. Without the fix,
    /// query_lower_tiers_with_options would re-query that worker from root in the lower tier,
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
