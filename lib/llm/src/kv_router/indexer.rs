// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Result;
use futures::StreamExt;

use dynamo_kv_router::{
    ConcurrentRadixTree, LowerTierIndexer, ThreadPoolIndexer,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{
        IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT, KvIndexer,
        KvIndexerInterface, KvIndexerMetrics, KvRouterError, LowerTierContinuation,
        LowerTierMatchDetails, MatchDetails,
    },
    protocols::{
        LocalBlockHash, OverlapScores, RouterEvent, StorageTier, TokensWithHashes, WorkerId,
        WorkerWithDpRank,
    },
};
use dynamo_runtime::{
    component::Component,
    pipeline::{ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
    traits::DistributedRuntimeProvider,
};
use tokio::sync::oneshot;

type LowerTierIndexers = Arc<Mutex<HashMap<StorageTier, Arc<LowerTierIndexer>>>>;

fn new_lower_tier_indexers() -> LowerTierIndexers {
    Arc::new(Mutex::new(HashMap::new()))
}

fn get_or_create_lower_tier_indexer(
    indexers: &LowerTierIndexers,
    storage_tier: StorageTier,
) -> Arc<LowerTierIndexer> {
    debug_assert!(!storage_tier.is_gpu());
    let mut lower_tier_indexers = indexers.lock().unwrap();
    lower_tier_indexers
        .entry(storage_tier)
        .or_insert_with(|| Arc::new(LowerTierIndexer::new()))
        .clone()
}

fn all_lower_tier_indexers(indexers: &LowerTierIndexers) -> Vec<Arc<LowerTierIndexer>> {
    let lower_tier_indexers = indexers.lock().unwrap();
    lower_tier_indexers.values().cloned().collect()
}

fn get_lower_tier_indexer(
    indexers: &LowerTierIndexers,
    storage_tier: StorageTier,
) -> Option<Arc<LowerTierIndexer>> {
    let lower_tier_indexers = indexers.lock().unwrap();
    lower_tier_indexers.get(&storage_tier).cloned()
}

fn lower_tier_query_order() -> [StorageTier; 3] {
    [
        StorageTier::HostPinned,
        StorageTier::Disk,
        StorageTier::External,
    ]
}

fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
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
        let Some(indexer) = get_lower_tier_indexer(indexers, storage_tier) else {
            continue;
        };

        for worker in indexer.workers() {
            continuations
                .entry(worker)
                .or_insert_with(|| LowerTierContinuation::from_root(0));
        }

        let tier_matches = indexer.query_match_details(sequence, &continuations);
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

#[derive(Debug, Clone, Default)]
pub(crate) struct TieredMatchDetails {
    pub device: MatchDetails,
    #[cfg_attr(not(test), allow(dead_code))]
    pub lower_tier: HashMap<StorageTier, LowerTierMatchDetails>,
}

pub struct RemoteIndexer {
    router: PushRouter<IndexerQueryRequest, IndexerQueryResponse>,
    model_name: String,
    namespace: String,
}

impl RemoteIndexer {
    async fn new(
        component: &Component,
        indexer_component_name: &str,
        model_name: String,
    ) -> Result<Self> {
        let namespace = component.namespace().name();
        let indexer_ns = component.namespace();
        let indexer_component = indexer_ns.component(indexer_component_name)?;
        let endpoint = indexer_component.endpoint(KV_INDEXER_QUERY_ENDPOINT);
        let client = endpoint.client().await?;
        let router =
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?;
        Ok(Self {
            router,
            model_name,
            namespace,
        })
    }

    async fn find_matches(&self, block_hashes: Vec<LocalBlockHash>) -> Result<OverlapScores> {
        let request = IndexerQueryRequest {
            model_name: self.model_name.clone(),
            namespace: self.namespace.clone(),
            block_hashes,
        };
        let mut stream: ManyOut<IndexerQueryResponse> =
            self.router.round_robin(SingleIn::new(request)).await?;

        match stream.next().await {
            Some(IndexerQueryResponse::Scores(scores)) => Ok(scores.into()),
            Some(IndexerQueryResponse::Error(msg)) => {
                Err(anyhow::anyhow!("Remote indexer error: {}", msg))
            }
            None => Err(anyhow::anyhow!("Remote indexer returned empty response")),
        }
    }
}

#[derive(Clone)]
pub enum Indexer {
    KvIndexer {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTree>>,
        lower_tier: LowerTierIndexers,
    },
    Remote(Arc<RemoteIndexer>),
    None,
}

impl Indexer {
    pub async fn new(
        component: &Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
        model_name: Option<String>,
    ) -> Result<Self> {
        if kv_router_config.overlap_score_weight == 0.0 {
            return Ok(Self::None);
        }

        if let Some(ref indexer_component_name) = kv_router_config.remote_indexer_component {
            let model_name = model_name.ok_or_else(|| {
                anyhow::anyhow!(
                    "model_name is required when remote_indexer_component is configured"
                )
            })?;
            tracing::info!(
                remote_indexer_component = %indexer_component_name,
                model_name,
                "Using remote KV indexer"
            );
            let remote = RemoteIndexer::new(component, indexer_component_name, model_name).await?;
            return Ok(Self::Remote(Arc::new(remote)));
        }

        if !kv_router_config.use_kv_events {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let cancellation_token = component.drt().primary_token();
            let prune_config = Some(PruneConfig {
                ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
                max_tree_size: kv_router_config.router_max_tree_size,
                prune_target_ratio: kv_router_config.router_prune_target_ratio,
            });
            return Ok(Self::KvIndexer {
                primary: KvIndexer::new_with_frequency(
                    cancellation_token,
                    None,
                    block_size,
                    kv_indexer_metrics,
                    prune_config,
                ),
                lower_tier: new_lower_tier_indexers(),
            });
        }

        if kv_router_config.router_event_threads > 1 {
            return Ok(Self::Concurrent {
                primary: Arc::new(ThreadPoolIndexer::new(
                    ConcurrentRadixTree::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                )),
                lower_tier: new_lower_tier_indexers(),
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
            lower_tier: new_lower_tier_indexers(),
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
        match self {
            Self::KvIndexer { primary, .. } => primary.find_match_details(sequence).await,
            Self::Concurrent { primary, .. } => {
                Ok(primary.backend().find_match_details_impl(&sequence, false))
            }
            Self::Remote(remote) => remote
                .find_matches(sequence)
                .await
                .map(|overlap_scores| MatchDetails {
                    overlap_scores,
                    ..Default::default()
                })
                .map_err(|e| {
                    tracing::warn!(error = %e, "Remote indexer query failed");
                    KvRouterError::IndexerOffline
                }),
            Self::None => Ok(MatchDetails::new()),
        }
    }

    pub(crate) async fn find_matches_by_tier(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<TieredMatchDetails, KvRouterError> {
        let device = self.find_match_details(sequence.clone()).await?;
        let lower_tier = match self {
            Self::KvIndexer { lower_tier, .. } | Self::Concurrent { lower_tier, .. } => {
                query_lower_tiers(lower_tier, &sequence, &device)
            }
            Self::Remote(_) | Self::None => HashMap::new(),
        };

        Ok(TieredMatchDetails { device, lower_tier })
    }

    pub(crate) async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => primary.dump_events().await,
            Self::Concurrent { primary, .. } => primary.dump_events().await,
            Self::Remote(_) => Ok(Vec::new()),
            Self::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_weight set to 0?)"
                );
            }
        }
    }

    pub(crate) async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => {
                primary
                    .process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Self::Concurrent { primary, .. } => {
                primary
                    .process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Self::Remote(_) | Self::None => Ok(()),
        }
    }

    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
            } => match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Cleared => {
                    if let Err(e) = primary.event_sender().send(event.clone()).await {
                        tracing::warn!("Failed to send event to indexer: {e}");
                    }

                    for indexer in all_lower_tier_indexers(lower_tier) {
                        if let Err(e) = indexer.apply_event(event.clone()) {
                            tracing::warn!(
                                error = %e,
                                "Failed to apply cleared event to lower-tier indexer"
                            );
                        }
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    if let Err(e) = primary.event_sender().send(event).await {
                        tracing::warn!("Failed to send event to indexer: {e}");
                    }
                }
                _ => {
                    if let Err(e) = get_or_create_lower_tier_indexer(lower_tier, event.storage_tier)
                        .apply_event(event)
                    {
                        tracing::warn!(error = %e, "Failed to apply event to lower-tier indexer");
                    }
                }
            },
            Self::Concurrent {
                primary,
                lower_tier,
            } => match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Cleared => {
                    primary.apply_event(event.clone()).await;

                    for indexer in all_lower_tier_indexers(lower_tier) {
                        if let Err(e) = indexer.apply_event(event.clone()) {
                            tracing::warn!(
                                error = %e,
                                "Failed to apply cleared event to lower-tier indexer"
                            );
                        }
                    }
                }
                _ if event.storage_tier.is_gpu() => {
                    primary.apply_event(event).await;
                }
                _ => {
                    if let Err(e) = get_or_create_lower_tier_indexer(lower_tier, event.storage_tier)
                        .apply_event(event)
                    {
                        tracing::warn!(error = %e, "Failed to apply event to lower-tier indexer");
                    }
                }
            },
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
            } => {
                for indexer in all_lower_tier_indexers(lower_tier) {
                    indexer.remove_worker(worker_id);
                }
                if let Err(e) = primary.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
            } => {
                for indexer in all_lower_tier_indexers(lower_tier) {
                    indexer.remove_worker(worker_id);
                }
                KvIndexerInterface::remove_worker(primary.as_ref(), worker_id).await;
            }
            Self::Remote(_) | Self::None => {}
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
            Self::Concurrent { primary, .. } => primary.backend().get_workers(),
            Self::Remote(_) | Self::None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::{Indexer, new_lower_tier_indexers};
    use dynamo_kv_router::{
        indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics},
        protocols::{
            ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
            KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier, WorkerWithDpRank,
            compute_seq_hash_for_block,
        },
    };

    fn make_test_indexer() -> Indexer {
        Indexer::KvIndexer {
            primary: KvIndexer::new(
                CancellationToken::new(),
                4,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            ),
            lower_tier: new_lower_tier_indexers(),
        }
    }

    async fn flush_primary(indexer: &Indexer) {
        match indexer {
            Indexer::KvIndexer { primary, .. } => {
                let _ = primary.flush().await;
            }
            Indexer::Concurrent { primary, .. } => {
                primary.flush().await;
            }
            Indexer::Remote(_) | Indexer::None => {}
        }
    }

    fn store_event(
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
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
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
        flush_primary(&indexer).await;

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
        flush_primary(&indexer).await;

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
}
