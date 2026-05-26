// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError, ThreadPoolIndexer},
    protocols::{DpRank, LocalBlockHash, OverlapScores, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::{component::Component, traits::DistributedRuntimeProvider};
use dynamo_tokens::SequenceHash;

#[derive(Clone)]
pub enum SideIndexer {
    KvIndexer(KvIndexer),
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTreeCompressed>>),
}

impl SideIndexer {
    pub(super) fn new_predict_on_route(
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

    pub(super) async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Self::Concurrent(indexer) => indexer.find_matches(sequence).await,
        }
    }

    pub(super) async fn process_routing_decision_with_hashes(
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

    pub(super) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Self::KvIndexer(indexer) => {
                KvIndexerInterface::remove_worker(indexer, worker_id).await;
            }
            Self::Concurrent(indexer) => {
                KvIndexerInterface::remove_worker(indexer.as_ref(), worker_id).await;
            }
        }
    }

    pub(super) async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
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
