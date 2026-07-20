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
        ThreadPoolIndexer,
    },
    protocols::{DpRank, RouterEvent, WorkerId},
};

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_router::indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_router::indexer::WireTieredMatchDetails;
use dynamo_runtime::component::Component;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

mod embedding_cache;
mod lookup;
mod recording;
mod recovery;
pub mod remote;
mod side;

pub use self::embedding_cache::{EmbeddingCacheIndexer, try_build_cache_indexer};
use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub use self::side::SideIndexer;
pub(crate) use recovery::{start_subscriber, start_worker_kv_query_endpoint};

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
        primary_records_routing_decisions: bool,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
        lower_tier: LowerTierIndexers,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    Remote {
        primary: Arc<RemoteIndexer>,
        approx: Option<SideIndexer>,
        primary_records_routing_decisions: bool,
    },
    None,
}

impl Indexer {
    pub(crate) fn supports_overlap_refresh(&self) -> bool {
        matches!(self, Self::KvIndexer { .. } | Self::Concurrent { .. })
    }

    pub(crate) fn supports_router_hint_chain_retention(&self) -> bool {
        matches!(
            self,
            Self::KvIndexer { approx: None, .. } | Self::Concurrent { approx: None, .. }
        )
    }

    pub async fn new(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<&str>,
        cancellation_token: CancellationToken,
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
            let approx = SideIndexer::new_predict_on_route(
                component,
                kv_router_config,
                block_size,
                cancellation_token.child_token(),
            );
            return Ok(Self::Remote {
                primary: Arc::new(remote),
                approx,
                primary_records_routing_decisions: !kv_router_config.use_kv_events,
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
                        Some(kv_indexer_metrics.clone()),
                        prune_config,
                    )),
                    lower_tier: LowerTierIndexers::new_with_metrics(
                        kv_router_config.router_event_threads as usize,
                        block_size,
                        Some(kv_indexer_metrics),
                    ),
                    approx: None,
                    primary_records_routing_decisions: true,
                });
            }

            return Ok(Self::KvIndexer {
                primary: KvIndexer::new_with_pruning(
                    cancellation_token.child_token(),
                    block_size,
                    kv_indexer_metrics.clone(),
                    prune_config,
                ),
                lower_tier: LowerTierIndexers::new_with_metrics(
                    1,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
                approx: None,
                primary_records_routing_decisions: true,
            });
        }

        let approx = SideIndexer::new_predict_on_route(
            component,
            kv_router_config,
            block_size,
            cancellation_token.child_token(),
        );

        if kv_router_config.router_event_threads > 1 {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            return Ok(Self::Concurrent {
                primary: Arc::new(ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics.clone()),
                )),
                lower_tier: LowerTierIndexers::new_with_metrics(
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
                approx,
                primary_records_routing_decisions: false,
            });
        }

        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        Ok(Self::KvIndexer {
            primary: KvIndexer::new_with_pruning(
                cancellation_token.child_token(),
                block_size,
                kv_indexer_metrics.clone(),
                None,
            ),
            lower_tier: LowerTierIndexers::new_with_metrics(
                1,
                block_size,
                Some(kv_indexer_metrics),
            ),
            approx,
            primary_records_routing_decisions: false,
        })
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
                ..
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
                ..
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
                ..
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
                ..
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
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, RoutingDecisionHashes},
        protocols::{
            BlockHashOptions, LocalBlockHash, StorageTier, TokensWithHashes, WorkerWithDpRank,
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
            .apply_event(store_event(
                10,
                0,
                1,
                &[],
                &[11, 12, 13],
                StorageTier::Device,
            ))
            .await;
        indexer
            .apply_event(store_event(
                10,
                0,
                2,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await;
        // Crucially, also give side_only_worker a HostPinned extension at
        // block 14 anchored on the same prefix [11, 12, 13]. If the lower
        // tier were seeded from the side-merged device score, the host walk
        // would find this and credit a hit; with the reorder it should not.
        indexer
            .apply_event(store_event(
                20,
                0,
                3,
                &[11, 12, 13],
                &[14],
                StorageTier::HostPinned,
            ))
            .await;

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
            .apply_event(store_event(10, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await;
        indexer
            .apply_event(store_event(
                10,
                0,
                2,
                &[11, 12],
                &[13],
                StorageTier::HostPinned,
            ))
            .await;

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
