// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{
        IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT, KvIndexer,
        KvIndexerInterface, KvIndexerMetrics, KvRouterError,
    },
    protocols::{
        LocalBlockHash, OverlapScores, RouterEvent, TokensWithHashes, WorkerId, WorkerWithDpRank,
    },
};
use dynamo_runtime::{
    component::Component,
    pipeline::{ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
    traits::DistributedRuntimeProvider,
};
use tokio::sync::oneshot;

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
            return Ok(Self::KvIndexer(KvIndexer::new_with_frequency(
                cancellation_token,
                None,
                block_size,
                kv_indexer_metrics,
                prune_config,
            )));
        }

        if kv_router_config.router_event_threads > 1 {
            return Ok(Self::Concurrent(Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                kv_router_config.router_event_threads as usize,
                block_size,
            ))));
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
            Self::Remote(remote) => remote.find_matches(sequence).await.map_err(|e| {
                tracing::warn!(error = %e, "Remote indexer query failed");
                KvRouterError::IndexerOffline
            }),
            Self::None => Ok(OverlapScores::new()),
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
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Self::Concurrent(tpi) => {
                tpi.process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Self::Remote(_) | Self::None => Ok(()),
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
