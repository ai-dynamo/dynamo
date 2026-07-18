// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed,
    approx::PruneConfig,
    config::KvRouterConfig,
    indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvRouterError, RoutingDecisionHashes,
        SyncIndexer, ThreadPoolIndexer,
    },
    protocols::{DpRank, OverlapScores, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::component::Component;
use tokio_util::sync::CancellationToken;

use super::lookup::HashInput;

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
        cancellation_token: CancellationToken,
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

        Some(Self::KvIndexer(KvIndexer::new_with_pruning(
            cancellation_token,
            block_size,
            metrics,
            prune_config,
        )))
    }

    pub(super) async fn find_matches_input(
        &self,
        sequence: HashInput<'_>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .find_matches(sequence.into_owned_at_boundary())
                    .await
            }
            Self::Concurrent(indexer) => {
                Ok(indexer.backend().find_matches(sequence.as_slice(), false))
            }
        }
    }

    pub(super) async fn process_routing_decision_hashes(
        &self,
        worker: WorkerWithDpRank,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_with_hashes(
                        worker,
                        hashes.local_hashes,
                        hashes.sequence_hashes,
                    )
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .process_routing_decision_hash_slices(
                        worker,
                        &hashes.local_hashes,
                        &hashes.sequence_hashes,
                    )
                    .await
            }
        }
    }

    pub(super) async fn reset_worker_dp_rank_and_wait(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Self::KvIndexer(indexer) => {
                indexer
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await
            }
            Self::Concurrent(indexer) => {
                indexer
                    .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
                    .await
            }
        }
    }

    pub(super) async fn flush_worker_and_wait(
        &self,
        worker_id: WorkerId,
    ) -> Result<(), KvRouterError> {
        match self {
            // The legacy indexer has one mutation lane, so its existing flush is already the
            // narrowest FIFO acknowledgement it can provide.
            Self::KvIndexer(indexer) => {
                indexer.flush_and_wait().await?;
            }
            Self::Concurrent(indexer) => {
                indexer.flush_worker_and_wait(worker_id).await?;
            }
        }
        Ok(())
    }
}
