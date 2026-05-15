// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError, LowerTierIndexers,
        MatchDetails, ThreadPoolIndexer, query_lower_tiers,
    },
    protocols::{
        DpRank, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId,
        WorkerWithDpRank,
    },
};

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_router::indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_router::indexer::WireTieredMatchDetails;
use dynamo_runtime::{component::Component, traits::DistributedRuntimeProvider};
use dynamo_tokens::SequenceHash;
use tokio::sync::oneshot;

mod jetstream;
pub mod remote;
mod subscriber;
mod worker_query;

use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub(crate) use subscriber::start_subscriber;
pub(crate) use worker_query::start_worker_kv_query_endpoint;

#[derive(Clone)]
pub enum SideIndexer {
    KvIndexer(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

impl SideIndexer {
    fn new_predict_on_route(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
    ) -> Option<Self> {
        let ttl_secs = kv_router_config.router_predicted_ttl_secs?;
        let prune_config = Some(PruneConfig {
            ttl: Duration::from_secs_f64(ttl_secs),
        });
        let metrics = KvIndexerMetrics::from_component(component);
        tracing::info!(
            ttl_secs,
            "Starting predict-on-route side indexer (short-TTL approximate)"
        );
        if kv_router_config.router_event_threads > 1 {
            return Some(Self::Concurrent(Arc::new(
                ThreadPoolIndexer::new_with_metrics_and_pruning(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(metrics),
                    prune_config,
                ),
            )));
        }

        let cancellation_token = component.drt().primary_token();
        Some(Self::KvIndexer(KvIndexer::new_with_frequency(
            cancellation_token,
            None,
            block_size,
            metrics,
            prune_config,
        )))
    }

    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Self::Concurrent(indexer) => indexer.find_matches(sequence).await,
        }
    }

    async fn process_routing_decision_with_hashes(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer(indexer) => {
                KvIndexerInterface::remove_worker(indexer, worker_id).await;
            }
            Self::Concurrent(indexer) => {
                KvIndexerInterface::remove_worker(indexer.as_ref(), worker_id).await;
            }
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        match self {
            Self::KvIndexer(indexer) => {
                KvIndexerInterface::remove_worker_dp_rank(indexer, worker_id, dp_rank).await;
            }
            Self::Concurrent(indexer) => {
                KvIndexerInterface::remove_worker_dp_rank(indexer.as_ref(), worker_id, dp_rank)
                    .await;
            }
        }
    }
}

/// `approx` is the optional predict-on-route side indexer. It is always local
/// to this router, even when the primary indexer is served or consumed
/// remotely. Routing decisions populate it with a short TTL; engine KV events
/// go to the primary only. `find_match_details` queries both and returns the
/// per-worker max overlap. Keeping this separate from the primary avoids the
/// sequence-hash mismatch problem: vLLM/SGLang salt their hashes with
/// cryptographic digests the router can't reproduce, so writing
/// router-computed hashes into the primary would key the same block under two
/// hashes and pollute the tree.
#[derive(Clone)]
pub enum Indexer {
    KvIndexer {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
    },
    Remote {
        primary: Arc<RemoteIndexer>,
        approx: Option<SideIndexer>,
    },
    None,
}

impl Indexer {
    pub async fn new(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<&str>,
    ) -> Result<Self> {
        if kv_router_config.overlap_score_credit == 0.0 {
            return Ok(Self::None);
        }

        if kv_router_config.router_predicted_ttl_secs.is_some() && !kv_router_config.use_kv_events {
            anyhow::bail!(
                "router_predicted_ttl_secs requires use_kv_events=true; \
                 do not combine a primary approximate indexer with a side approximate indexer"
            );
        }

        if kv_router_config.use_remote_indexer {
            let model_name = model_name
                .ok_or_else(|| {
                    anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
                })?
                .to_string();
            let indexer_component_name = component.name();
            tracing::info!(
                indexer_component = %indexer_component_name,
                model_name,
                "Using remote KV indexer"
            );
            let remote =
                RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
            let approx = SideIndexer::new_predict_on_route(component, kv_router_config, block_size);
            return Ok(Self::Remote {
                primary: Arc::new(remote),
                approx,
            });
        }

        if !kv_router_config.use_kv_events {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let prune_config = Some(PruneConfig {
                ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
            });
            if kv_router_config.router_event_threads > 1 {
                return Ok(Self::Concurrent {
                    primary: Arc::new(ThreadPoolIndexer::new_with_metrics_and_pruning(
                        ConcurrentRadixTreeCompressed::new(),
                        kv_router_config.router_event_threads as usize,
                        block_size,
                        Some(kv_indexer_metrics),
                        prune_config,
                    )),
                    lower_tier: LowerTierIndexers::new(
                        kv_router_config.router_event_threads as usize,
                        block_size,
                    ),
                    approx: None,
                });
            }

            let cancellation_token = component.drt().primary_token();
            return Ok(Self::KvIndexer {
                primary: KvIndexer::new_with_frequency(
                    cancellation_token,
                    None,
                    block_size,
                    kv_indexer_metrics,
                    prune_config,
                ),
                lower_tier: LowerTierIndexers::new(1, block_size),
                approx: None,
            });
        }

        let approx = SideIndexer::new_predict_on_route(component, kv_router_config, block_size);

        if kv_router_config.router_event_threads > 1 {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            return Ok(Self::Concurrent {
                primary: Arc::new(ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics),
                )),
                lower_tier: LowerTierIndexers::new(
                    kv_router_config.router_event_threads as usize,
                    block_size,
                ),
                approx,
            });
        }

        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let cancellation_token = component.drt().primary_token();

        Ok(Self::KvIndexer {
            primary: KvIndexer::new_with_frequency(
                cancellation_token,
                None,
                block_size,
                kv_indexer_metrics,
                None,
            ),
            lower_tier: LowerTierIndexers::new(1, block_size),
            approx,
        })
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
        let primary_details = self.find_primary_match_details(sequence.clone()).await?;
        let approx = match self {
            Self::KvIndexer { approx, .. }
            | Self::Concurrent { approx, .. }
            | Self::Remote { approx, .. } => approx.as_ref(),
            Self::None => None,
        };

        let Some(approx) = approx else {
            return Ok(primary_details);
        };
        match approx.find_matches(sequence).await {
            Ok(side) => Ok(merge_overlap_scores(primary_details, side)),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "predict-on-route side indexer query failed; using primary only"
                );
                Ok(primary_details)
            }
        }
    }

    pub(crate) async fn find_primary_match_details(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<MatchDetails, KvRouterError> {
        let primary_details = match self {
            Self::KvIndexer { primary, .. } => primary.find_match_details(sequence.clone()).await?,
            Self::Concurrent { primary, .. } => {
                primary.backend().find_match_details_impl(&sequence, false)
            }
            Self::Remote { primary, .. } => {
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

    pub(crate) async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self {
            Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                let device = self.find_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            Self::Remote { primary, approx } => {
                let mut tiered = primary
                    .find_matches_by_tier(sequence.clone(), false)
                    .await
                    .map_err(|e| {
                        tracing::warn!(error = %e, "Remote indexer tiered query failed");
                        KvRouterError::IndexerOffline
                    })?;
                if let Some(approx) = approx {
                    match approx.find_matches(sequence).await {
                        Ok(side) => {
                            tiered.device = merge_overlap_scores(tiered.device, side);
                        }
                        Err(error) => {
                            tracing::warn!(
                                error = %error,
                                "predict-on-route side indexer query failed; using remote primary only"
                            );
                        }
                    }
                }
                Ok(tiered)
            }
            Self::None => Ok(TieredMatchDetails::default()),
        }
    }

    pub(crate) async fn find_primary_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        match self {
            Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                let device = self.find_primary_match_details(sequence.clone()).await?;
                let lt = query_lower_tiers(lower_tier, &sequence, &device);
                Ok(TieredMatchDetails {
                    device,
                    lower_tier: lt,
                })
            }
            Self::Remote { primary, .. } => primary
                .find_matches_by_tier(sequence.clone(), false)
                .await
                .map_err(|e| {
                    tracing::warn!(error = %e, "Remote indexer tiered query failed");
                    KvRouterError::IndexerOffline
                }),
            Self::None => Ok(TieredMatchDetails::default()),
        }
    }

    pub(crate) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        if let Self::Remote {
            primary,
            approx: Some(approx),
        } = self
        {
            debug_assert!(
                primary.use_kv_events(),
                "remote side indexer requires an event-driven primary"
            );
            return approx
                .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                .await;
        }

        if let Some(approx) = match self {
            Self::KvIndexer { approx, .. } | Self::Concurrent { approx, .. } => approx.as_ref(),
            _ => None,
        } {
            return approx
                .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                .await;
        }

        match self {
            Self::KvIndexer { primary, .. } => {
                primary
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent { primary, .. } => {
                primary
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Remote { primary, .. } => primary
                .record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                .await
                .map_err(|error| {
                    tracing::warn!(error = %error, "Remote indexer write failed");
                    KvRouterError::IndexerDroppedRequest
                }),
            Self::None => Ok(()),
        }
    }

    pub(crate) async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => primary.dump_events().await,
            Self::Concurrent { primary, .. } => primary.dump_events().await,
            Self::Remote { .. } => Ok(Vec::new()),
            Self::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_credit set to 0?)"
                );
            }
        }
    }

    pub(crate) async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // The Concurrent primary doesn't accept hashed writes, so when there's
        // no side indexer we hand the request to its native (non-hashed) path.
        // Every other case routes through `record_hashed_routing_decision`,
        // which dispatches to the side indexer when enabled and otherwise to
        // the appropriate primary.
        if let Self::Concurrent {
            primary,
            approx: None,
            ..
        } = self
        {
            return primary
                .process_routing_decision_for_request(tokens_with_hashes, worker)
                .await;
        }
        match self {
            Self::KvIndexer { .. } | Self::Concurrent { .. } | Self::Remote { .. } => {
                let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                self.record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::None => Ok(()),
        }
    }

    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                ..
            } => match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Cleared => {
                    if let Err(e) = primary.event_sender().send(event.clone()).await {
                        tracing::warn!("Failed to send event to indexer: {e}");
                    }

                    for indexer in lower_tier.all() {
                        indexer.apply_event(event.clone()).await;
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    if let Err(e) = primary.event_sender().send(event).await {
                        tracing::warn!("Failed to send event to indexer: {e}");
                    }
                }
                _ => {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            },
            Self::Concurrent {
                primary,
                lower_tier,
                ..
            } => match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Cleared => {
                    primary.apply_event(event.clone()).await;

                    for indexer in lower_tier.all() {
                        indexer.apply_event(event.clone()).await;
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    primary.apply_event(event).await;
                }
                _ => {
                    lower_tier
                        .get_or_create(event.storage_tier)
                        .apply_event(event)
                        .await;
                }
            },
            Self::Remote { .. } | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                approx,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                if let Err(e) = primary.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
                if let Some(approx) = approx {
                    approx.remove_worker(worker_id).await;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                approx,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                KvIndexerInterface::remove_worker(primary.as_ref(), worker_id).await;
                if let Some(approx) = approx {
                    approx.remove_worker(worker_id).await;
                }
            }
            Self::Remote { approx, .. } => {
                if let Some(approx) = approx {
                    approx.remove_worker(worker_id).await;
                }
            }
            Self::None => {}
        }
    }

    pub(crate) async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
                approx,
            } => {
                for indexer in lower_tier.all() {
                    KvIndexerInterface::remove_worker_dp_rank(&*indexer, worker_id, dp_rank).await;
                }
                KvIndexerInterface::remove_worker_dp_rank(primary, worker_id, dp_rank).await;
                if let Some(approx) = approx {
                    approx.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
                approx,
            } => {
                for indexer in lower_tier.all() {
                    KvIndexerInterface::remove_worker_dp_rank(&*indexer, worker_id, dp_rank).await;
                }
                KvIndexerInterface::remove_worker_dp_rank(primary.as_ref(), worker_id, dp_rank)
                    .await;
                if let Some(approx) = approx {
                    approx.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
            }
            Self::Remote { approx, .. } => {
                if let Some(approx) = approx {
                    approx.remove_worker_dp_rank(worker_id, dp_rank).await;
                }
            }
            Self::None => {}
        }
    }

    pub(crate) async fn get_workers(&self) -> Vec<WorkerId> {
        match self {
            Self::KvIndexer { primary, .. } => {
                let (resp_tx, resp_rx) = oneshot::channel();
                let req = dynamo_kv_router::indexer::GetWorkersRequest { resp: resp_tx };
                if let Err(e) = primary.get_workers_sender().send(req).await {
                    tracing::warn!("Failed to send get_workers request: {e}");
                    return Vec::new();
                }
                resp_rx.await.unwrap_or_default()
            }
            Self::Concurrent { primary, .. } => primary.get_workers().await,
            Self::Remote { .. } | Self::None => Vec::new(),
        }
    }
}

/// Merge a side-indexer's `OverlapScores` into the primary's `MatchDetails`
/// by taking the per-worker max overlap. The side indexer covers the window
/// before the engine's first KV event arrives; for workers it knows about,
/// we use whichever indexer saw the longer prefix. `last_matched_hashes`,
/// `frequencies`, and `tree_sizes` come from the primary — the side
/// indexer's short-TTL view isn't meaningful for those signals.
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

#[cfg(test)]
pub(super) mod test_util {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    pub(crate) fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use super::test_util::store_event;
    use super::{Indexer, LowerTierIndexers};
    use dynamo_kv_router::{
        ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
        approx::PruneConfig,
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics},
        protocols::{
            BlockHashOptions, LocalBlockHash, StorageTier, WorkerWithDpRank,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
    };

    fn make_test_indexer() -> Indexer {
        Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: LowerTierIndexers::new(1, 4),
            approx: None,
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
        }
    }

    async fn flush_indexer(indexer: &Indexer) {
        match indexer {
            Indexer::KvIndexer {
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
    async fn tiered_query_chains_device_host_and_disk() {
        let indexer = make_test_indexer();
        let worker = WorkerWithDpRank::new(7, 0);

        indexer
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::Disk,
            ))
            .await;
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
    async fn tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores() {
        let indexer = make_test_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await;
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
            .apply_event(store_event(20, 0, 1, &[], &[31], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(21, 0, 2, &[], &[32], StorageTier::HostPinned))
            .await;
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
            .apply_event(store_event(7, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;
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
    async fn concurrent_tiered_query_seeds_lower_tier_only_workers_without_affecting_device_scores()
    {
        let indexer = make_test_concurrent_indexer();
        let device_worker = WorkerWithDpRank::new(10, 0);
        let host_only_worker = WorkerWithDpRank::new(20, 0);
        let disk_only_worker = WorkerWithDpRank::new(30, 0);

        indexer
            .apply_event(store_event(10, 0, 1, &[], &[21], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(20, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await;
        indexer
            .apply_event(store_event(30, 0, 3, &[], &[21], StorageTier::Disk))
            .await;
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

        // Worker has the same blocks in both device and host-pinned storage.
        indexer
            .apply_event(store_event(
                7,
                0,
                1,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await;
        indexer
            .apply_event(store_event(
                7,
                0,
                2,
                &[],
                &[11, 12, 13],
                StorageTier::HostPinned,
            ))
            .await;
        flush_indexer(&indexer).await;

        let matches = indexer
            .find_matches_by_tier(vec![
                LocalBlockHash(11),
                LocalBlockHash(12),
                LocalBlockHash(13),
            ])
            .await
            .unwrap();

        // Device overlap should be 3 blocks.
        assert_eq!(matches.device.overlap_scores.scores.get(&worker), Some(&3));

        // Lower-tier must NOT report additional hits for the same worker
        // whose blocks are already fully accounted for in the device tier.
        let host_hits = matches
            .lower_tier
            .get(&StorageTier::HostPinned)
            .and_then(|tier| tier.hits.get(&worker).copied())
            .unwrap_or(0);
        assert_eq!(
            host_hits, 0,
            "lower-tier should not double-count blocks already matched in device tier \
             (got {host_hits} host-pinned hits for a worker with full device overlap)"
        );
    }

    #[tokio::test]
    async fn concurrent_remove_worker_removes_lower_tier_state() {
        let indexer = make_test_concurrent_indexer();
        let worker = WorkerWithDpRank::new(20, 0);

        indexer
            .apply_event(store_event(20, 0, 1, &[], &[31], StorageTier::HostPinned))
            .await;
        flush_indexer(&indexer).await;

        let before = indexer
            .find_matches_by_tier(vec![LocalBlockHash(31)])
            .await
            .unwrap();
        assert_eq!(
            before
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|tier| tier.hits.get(&worker)),
            Some(&1)
        );

        indexer.remove_worker(20).await;
        flush_indexer(&indexer).await;

        let after = indexer
            .find_matches_by_tier(vec![LocalBlockHash(31)])
            .await
            .unwrap();
        assert!(
            !after
                .lower_tier
                .get(&StorageTier::HostPinned)
                .is_some_and(|tier| tier.hits.contains_key(&worker))
        );
    }
}
