// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError},
    protocols::{
        DpRank, LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId,
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

use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub(crate) use subscriber::start_subscriber;
pub(crate) use worker_query::start_worker_kv_query_endpoint;

#[derive(Clone)]
pub enum Indexer {
    KvIndexer(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
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

        // Approximate (indexer-only) mode and predict-on-route both rely on the
        // single-threaded KvIndexer's prune manager to record speculative
        // routing decisions. Without a PruneConfig, routing-decision messages
        // are silently dropped by the indexer task.
        let needs_prune_config =
            !kv_router_config.use_kv_events || kv_router_config.router_predict_on_route;

        if needs_prune_config {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            let cancellation_token = component.drt().primary_token();
            let prune_config = Some(PruneConfig {
                ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
                max_tree_size: kv_router_config.router_max_tree_size,
                prune_target_ratio: kv_router_config.router_prune_target_ratio,
            });
            return Ok(Self::KvIndexer(KvIndexer::new_with_frequency(
                cancellation_token,
                None,
                block_size,
                kv_indexer_metrics,
                prune_config,
            )));
        }

        if kv_router_config.router_event_threads > 1 {
            let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
            return Ok(Self::Concurrent(Arc::new(
                ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics),
                ),
            )));
        }

        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let cancellation_token = component.drt().primary_token();

        Ok(Self::KvIndexer(KvIndexer::new_with_frequency(
            cancellation_token,
            None,
            block_size,
            kv_indexer_metrics,
            None,
        )))
    }

    pub(crate) async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Self::Concurrent(tpi) => tpi.find_matches(sequence).await,
            Self::Remote(remote) => match remote.find_matches(sequence).await {
                Ok(scores) => Ok(scores),
                Err(error) => {
                    tracing::warn!(error = %error, "Remote indexer query failed");
                    Ok(OverlapScores::new())
                }
            },
            Self::None => Ok(OverlapScores::new()),
        }
    }

    pub(crate) async fn record_hashed_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
        ttl_override: Option<Duration>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(
                        worker,
                        local_hashes,
                        sequence_hashes,
                        ttl_override,
                    )
                    .await
            }
            Self::Concurrent(_) => {
                tracing::warn!(
                    "Hashed routing-decision recording is unsupported for concurrent indexers"
                );
                Err(KvRouterError::IndexerDroppedRequest)
            }
            Self::Remote(remote) => remote
                .record_hashed_routing_decision(worker, local_hashes, sequence_hashes, ttl_override)
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
            Self::KvIndexer(indexer) => indexer.dump_events().await,
            Self::Concurrent(tpi) => tpi.dump_events().await,
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
        ttl_override: Option<Duration>,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(_) | Self::Remote(_) => {
                let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
                let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
                self.record_hashed_routing_decision(
                    worker,
                    local_hashes,
                    sequence_hashes,
                    ttl_override,
                )
                .await
            }
            Self::Concurrent(tpi) => {
                tpi.process_routing_decision_for_request(tokens_with_hashes, worker, ttl_override)
                    .await
            }
            Self::None => Ok(()),
        }
    }

    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        match self {
            Self::KvIndexer(indexer) => {
                if let Err(e) = indexer.event_sender().send(event).await {
                    tracing::warn!("Failed to send event to indexer: {e}");
                }
            }
            Self::Concurrent(tpi) => tpi.apply_event(event).await,
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer(indexer) => {
                if let Err(e) = indexer.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
            }
            Self::Concurrent(tpi) => {
                KvIndexerInterface::remove_worker(tpi.as_ref(), worker_id).await;
            }
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        match self {
            Self::KvIndexer(indexer) => {
                KvIndexerInterface::remove_worker_dp_rank(indexer, worker_id, dp_rank).await;
            }
            Self::Concurrent(tpi) => {
                KvIndexerInterface::remove_worker_dp_rank(tpi.as_ref(), worker_id, dp_rank).await;
            }
            Self::Remote(_) | Self::None => {}
        }
    }

    pub(crate) async fn get_workers(&self) -> Vec<WorkerId> {
        match self {
            Self::KvIndexer(indexer) => {
                let (resp_tx, resp_rx) = oneshot::channel();
                let req = dynamo_kv_router::indexer::GetWorkersRequest { resp: resp_tx };
                if let Err(e) = indexer.get_workers_sender().send(req).await {
                    tracing::warn!("Failed to send get_workers request: {e}");
                    return Vec::new();
                }
                resp_rx.await.unwrap_or_default()
            }
            Self::Concurrent(tpi) => tpi.backend().get_workers(),
            Self::Remote(_) | Self::None => Vec::new(),
        }
    }
}
