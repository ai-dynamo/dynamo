// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a core-owned KV index is fed with worker KV events.

use async_trait::async_trait;

use super::error::SelectionError;
use super::types::WorkerCatalogRecord;
use crate::identity::RoutingPartitionId;
use crate::services::indexer::backend::Indexer;
use crate::services::indexer::registry::WorkerRegistry;

/// Builds and feeds a partition's index ([`super::KvIndexSource::Owned`]).
///
/// The registry holds the standalone indexes and the per-rank listener
/// lifecycle; the ZMQ ingress works on it. A host with its own event transport
/// (the frontend over the Dynamo runtime) builds and feeds the index itself
/// and treats the per-worker hooks as membership it already tracks.
#[async_trait]
pub trait KvEventIngress: Send + Sync {
    /// The index for `key`, created on first use. Called once per partition,
    /// before any worker attaches.
    fn open(&self, registry: &WorkerRegistry, key: &RoutingPartitionId, block_size: u32)
    -> Indexer;

    /// Metadata `record` must carry before this ingress can feed its ranks.
    /// A non-empty result keeps the worker `Incomplete`. Consulted only when
    /// the core subscribes to worker KV events.
    fn missing_metadata(&self, record: &WorkerCatalogRecord) -> Vec<String> {
        let _ = record;
        Vec::new()
    }

    /// Start feeding every rank of `record` into its partition's index.
    /// Called only when the core subscribes to worker KV events.
    async fn attach(
        &self,
        registry: &WorkerRegistry,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let _ = (registry, record);
        Ok(())
    }

    /// `record` left the catalog: stop feeding it and drop what the index
    /// holds for it. Called for every worker, subscribed or not.
    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
        let _ = (registry, record);
    }
}

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
        record
            .dp_ranks()
            .filter(|rank| {
                endpoints
                    .get(rank)
                    .is_none_or(|endpoint| endpoint.is_empty())
            })
            .map(|rank| format!("kv_events endpoint is required for dp_rank {rank}"))
            .collect()
    }

    async fn attach(
        &self,
        registry: &WorkerRegistry,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        let mut endpoints: Vec<_> = record.listener_endpoints().into_iter().collect();
        endpoints.sort_by_key(|(dp_rank, _)| *dp_rank);
        for (dp_rank, endpoint) in endpoints {
            crate::services::common::zmq::validate_endpoint(&endpoint).map_err(|error| {
                SelectionError::BadRequest(format!(
                    "invalid kv_events endpoint for worker {} dp_rank {dp_rank}: {error}",
                    record.worker_id
                ))
            })?;
            if let Some(replay_endpoint) = record.replay_endpoint.as_deref() {
                crate::services::common::zmq::validate_endpoint(replay_endpoint).map_err(
                    |error| {
                        SelectionError::BadRequest(format!(
                            "invalid replay endpoint for worker {} dp_rank {dp_rank}: {error}",
                            record.worker_id
                        ))
                    },
                )?;
            }
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
        // No listeners (events disabled or a remote primary): drop what an
        // approximate or side index recorded for the worker.
        if let Some(indexer) = registry
            .get_indexer(&record.key())
            .map(|entry| entry.indexer.clone())
        {
            indexer.remove_worker(record.worker_id).await;
        }
    }
}
