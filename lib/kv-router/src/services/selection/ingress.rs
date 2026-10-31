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

    /// Reconcile event sources, retaining unchanged ranks and their cached state.
    /// `previous` is the last schedulable record in the same partition.
    /// Called only when the core subscribes to worker KV events.
    async fn reconcile(
        &self,
        registry: &WorkerRegistry,
        previous: Option<&WorkerCatalogRecord>,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let _ = (registry, previous, record);
        Ok(())
    }

    /// `record` left the catalog: stop feeding it and drop what the index
    /// holds for it. Called for every worker, subscribed or not.
    async fn detach(&self, registry: &WorkerRegistry, record: &WorkerCatalogRecord) {
        let _ = (registry, record);
    }
}

/// The kernel's own ingress: it opens a partition's index and nothing else. No
/// transport feeds the index; a host that listens for worker KV events
/// installs its ingress on `HostCache::index` instead.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocalIndexIngress;

impl KvEventIngress for LocalIndexIngress {
    fn open(
        &self,
        registry: &WorkerRegistry,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Indexer {
        registry.get_or_create_indexer(key.clone(), block_size)
    }
}
