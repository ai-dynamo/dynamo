// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a core-owned KV index is fed with worker KV events.

use async_trait::async_trait;

use super::error::SelectionError;
use super::types::WorkerCatalogRecord;
use crate::services::indexer::registry::WorkerRegistry;

/// Feeds a partition's core-owned index ([`super::KvIndexSource::Owned`]).
///
/// The registry holds the partition indexes and the per-rank listener
/// lifecycle; an ingress attaches and detaches workers on it. A host that feeds
/// the index itself uses [`super::KvIndexSource::Provided`] instead.
#[async_trait]
pub trait KvEventIngress: Send + Sync {
    /// Metadata `record` must carry before this ingress can feed its ranks.
    /// A non-empty result keeps the worker `Incomplete`.
    fn missing_metadata(&self, record: &WorkerCatalogRecord) -> Vec<String>;

    /// Start feeding every rank of `record` into its partition's index.
    async fn attach(
        &self,
        registry: &WorkerRegistry,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError>;

    /// Stop feeding `record` and drop its ranks from the index.
    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord);
}

/// One ZMQ SUB socket per worker rank on the engine's KV-event publisher, with
/// gap replay from the engine's replay endpoint when it advertises one.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZmqDirectIngress;

#[async_trait]
impl KvEventIngress for ZmqDirectIngress {
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
    }
}
