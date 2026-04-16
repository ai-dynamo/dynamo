// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, LowerTierIndexer, ThreadPoolIndexer,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError,
        LowerTierContinuation, LowerTierMatchDetails, MatchDetails,
    },
    protocols::{
        DpRank, LocalBlockHash, OverlapScores, RouterEvent, StorageTier, TokensWithHashes, WorkerId,
        WorkerWithDpRank,
    },
};
use dynamo_runtime::{component::Component, traits::DistributedRuntimeProvider};
use dynamo_tokens::SequenceHash;
use tokio::sync::oneshot;

mod jetstream;
pub mod remote;
mod subscriber;
mod worker_query;

#[derive(Clone)]
pub struct LowerTierIndexers {
    num_threads: usize,
    block_size: u32,
    indexers: Arc<Mutex<HashMap<StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>>>>,
}

impl LowerTierIndexers {
    pub(crate) fn new(num_threads: usize, block_size: u32) -> Self {
        assert!(
            num_threads > 0,
            "lower-tier indexer threads must be non-zero"
        );
        Self {
            num_threads,
            block_size,
            indexers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_or_create(&self, storage_tier: StorageTier) -> Arc<ThreadPoolIndexer<LowerTierIndexer>> {
        debug_assert!(!storage_tier.is_gpu());
        let mut lower_tier_indexers = self.indexers.lock().unwrap();
        lower_tier_indexers
            .entry(storage_tier)
            .or_insert_with(|| {
                Arc::new(ThreadPoolIndexer::new(
                    LowerTierIndexer::new(),
                    self.num_threads,
                    self.block_size,
                ))
            })
            .clone()
    }

    fn all(&self) -> Vec<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        let lower_tier_indexers = self.indexers.lock().unwrap();
        lower_tier_indexers.values().cloned().collect()
    }

    fn get(&self, storage_tier: StorageTier) -> Option<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        let lower_tier_indexers = self.indexers.lock().unwrap();
        lower_tier_indexers.get(&storage_tier).cloned()
    }
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
        let Some(indexer) = indexers.get(storage_tier) else {
            continue;
        };

        let continuations_before = continuations.len();
        if let Some(&first_hash) = sequence.first() {
            let root_workers: Vec<_> = indexer.backend().root_workers(first_hash);
            let mut new_roots = 0usize;
            for worker in root_workers.iter() {
                if !continuations.contains_key(worker) {
                    new_roots += 1;
                }
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

#[derive(Debug, Clone, Default)]
pub(crate) struct TieredMatchDetails {
    pub device: MatchDetails,
    #[cfg_attr(not(test), allow(dead_code))]
    pub lower_tier: HashMap<StorageTier, LowerTierMatchDetails>,
}

use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub(crate) use subscriber::start_subscriber;
pub(crate) use worker_query::start_worker_kv_query_endpoint;

#[derive(Clone)]
pub enum Indexer {
    KvIndexer {
        primary: KvIndexer,
        lower_tier: LowerTierIndexers,
    },
    Concurrent {
        primary: Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>,
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
        model_name: Option<&str>,
    ) -> Result<Self> {
        if kv_router_config.overlap_score_weight == 0.0 {
            return Ok(Self::None);
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
                lower_tier: LowerTierIndexers::new(1, block_size),
            });
        }

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

    pub(crate) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer { primary, .. } => {
                primary
                    .process_routing_decision_with_hashes(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent { .. } => {
                tracing::warn!(
                    "Hashed routing-decision recording is unsupported for concurrent indexers"
                );
                Err(KvRouterError::IndexerDroppedRequest)
            }
            Self::Remote(remote) => remote
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
            Self::KvIndexer { .. } | Self::Remote(_) => {
                let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                self.record_hashed_routing_decision(worker, local_hashes, sequence_hashes)
                    .await
            }
            Self::Concurrent { primary, .. } => {
                primary
                    .process_routing_decision_for_request(tokens_with_hashes, worker)
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
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                if let Err(e) = primary.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
            }
            Self::Concurrent {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    indexer.remove_worker(worker_id).await;
                }
                KvIndexerInterface::remove_worker(primary.as_ref(), worker_id).await;
            }
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        match self {
            Self::KvIndexer {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    KvIndexerInterface::remove_worker_dp_rank(&*indexer, worker_id, dp_rank).await;
                }
                KvIndexerInterface::remove_worker_dp_rank(primary, worker_id, dp_rank).await;
            }
            Self::Concurrent {
                primary,
                lower_tier,
            } => {
                for indexer in lower_tier.all() {
                    KvIndexerInterface::remove_worker_dp_rank(&*indexer, worker_id, dp_rank).await;
                }
                KvIndexerInterface::remove_worker_dp_rank(primary.as_ref(), worker_id, dp_rank)
                    .await;
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

    use super::{Indexer, LowerTierIndexers};
    use dynamo_kv_router::{
        ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
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
            lower_tier: LowerTierIndexers::new(1, 4),
        }
    }

    fn make_test_concurrent_indexer() -> Indexer {
        Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 4)),
            lower_tier: LowerTierIndexers::new(2, 4),
        }
    }

    async fn flush_indexer(indexer: &Indexer) {
        match indexer {
            Indexer::KvIndexer {
                primary,
                lower_tier,
            } => {
                let _ = primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
            }
            Indexer::Concurrent {
                primary,
                lower_tier,
            } => {
                primary.flush().await;
                for indexer in lower_tier.all() {
                    let _ = indexer.dump_events().await.unwrap();
                }
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
            .apply_event(store_event(7, 0, 1, &[], &[11, 12, 13], StorageTier::Device))
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
