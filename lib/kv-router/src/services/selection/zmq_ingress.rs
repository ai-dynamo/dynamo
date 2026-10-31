// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The ZMQ-direct ingress: workers publish KV events to endpoints the catalog
//! record names, and the worker registry runs a listener per data-parallel rank.

use async_trait::async_trait;

use super::error::SelectionError;
use super::ingress::KvEventIngress;
use super::types::WorkerCatalogRecord;
use crate::identity::RoutingPartitionId;
use crate::services::indexer::backend::Indexer;
use crate::services::indexer::registry::WorkerRegistry;

/// One ZMQ SUB socket per worker rank on the engine's KV-event publisher, with
/// gap replay from the engine's replay endpoint when it advertises one.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZmqDirectIngress;

#[async_trait]
impl KvEventIngress for ZmqDirectIngress {
    fn open(
        &self,
        registry: &WorkerRegistry,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Indexer {
        registry.get_or_create_indexer(key.clone(), block_size)
    }

    fn missing_metadata(&self, record: &WorkerCatalogRecord) -> Vec<String> {
        let endpoints = record.listener_endpoints();
        let mut missing: Vec<String> = record
            .dp_ranks()
            .filter(|rank| {
                endpoints
                    .get(rank)
                    .is_none_or(|endpoint| endpoint.is_empty())
            })
            .map(|rank| format!("kv_events endpoint is required for dp_rank {rank}"))
            .collect();
        // The replay protocol carries no rank, and each rank opens its own socket.
        if record.dp_size() > 1 && record.replay_endpoint.is_some() {
            missing.push("replay_endpoint is supported only for data_parallel_size 1".to_string());
        }
        missing
    }

    async fn reconcile(
        &self,
        registry: &WorkerRegistry,
        previous: Option<&WorkerCatalogRecord>,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        let endpoints = record.listener_endpoints();
        for (rank, endpoint) in &endpoints {
            crate::services::common::zmq::validate_endpoint(endpoint).map_err(|error| {
                SelectionError::BadRequest(format!(
                    "invalid kv_events endpoint for worker {} dp_rank {rank}: {error}",
                    record.worker_id
                ))
            })?;
        }
        if let Some(endpoint) = record.replay_endpoint.as_deref() {
            crate::services::common::zmq::validate_endpoint(endpoint).map_err(|error| {
                SelectionError::BadRequest(format!(
                    "invalid replay endpoint for worker {}: {error}",
                    record.worker_id
                ))
            })?;
        }
        let ranks: std::collections::HashSet<_> = record.dp_ranks().collect();
        let mut retained = std::collections::HashSet::new();
        if let Some(previous) = previous {
            let old_endpoints = previous.listener_endpoints();
            for rank in previous.dp_ranks() {
                // A schedulable record can lack its listener when an earlier
                // update was cancelled mid-deregistration.
                let listener_present = registry.has_listener(record.worker_id, rank);
                if listener_present
                    && ranks.contains(&rank)
                    && old_endpoints.get(&rank) == endpoints.get(&rank)
                    && previous.replay_endpoint == record.replay_endpoint
                {
                    retained.insert(rank);
                } else if listener_present {
                    registry
                        .deregister_dp_rank(
                            record.worker_id,
                            rank,
                            &previous.model_name,
                            &previous.routing_group,
                        )
                        .await
                        .map_err(|error| SelectionError::Internal(error.to_string()))?;
                } else {
                    registry
                        .remove_dp_rank_blocks(record.worker_id, rank, &previous.key())
                        .await;
                }
            }
        }
        let mut endpoints: Vec<_> = endpoints
            .into_iter()
            .filter(|(rank, _)| ranks.contains(rank) && !retained.contains(rank))
            .collect();
        endpoints.sort_by_key(|(rank, _)| *rank);
        for (dp_rank, endpoint) in endpoints {
            registry
                .register(
                    record.worker_id,
                    endpoint,
                    dp_rank,
                    record.model_name.clone(),
                    record.routing_group.clone(),
                    block_size,
                    record.replay_endpoint.clone(),
                )
                .await
                .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        }
        Ok(())
    }

    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
        if registry.has_worker(record.worker_id) {
            if let Err(error) = registry
                .deregister(record.worker_id, &record.model_name, &record.routing_group)
                .await
            {
                tracing::debug!(
                    worker_id = record.worker_id,
                    error = %error,
                    "indexer deregistration skipped or failed"
                );
            }
            return;
        }
        remove_worker_from_index(registry, record).await;
    }
}

/// No listeners for `record` (events disabled or a remote primary): drop what
/// an approximate or side index recorded for the worker.
async fn remove_worker_from_index(registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
    if let Some(indexer) = registry
        .get_indexer(&record.key())
        .map(|entry| entry.indexer.clone())
    {
        indexer.remove_worker(record.worker_id).await;
    }
}
