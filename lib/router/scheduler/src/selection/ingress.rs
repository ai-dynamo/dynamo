// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a core-owned KV index is fed with worker KV events.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::error::SelectionError;
use super::types::WorkerCatalogRecord;
use crate::identity::RoutingPartitionId;
use dynamo_kv_indexer::backend::Indexer;
use dynamo_kv_indexer::registry::IndexRegistry;

/// Builds and feeds a partition's index ([`super::KvIndexSource::Owned`]).
///
/// The registry holds the standalone indexes and the per-rank listener
/// lifecycle; the ZMQ ingress works on it. A host with its own event transport
/// (the frontend over the Dynamo runtime) builds and feeds the index itself
/// and treats the per-worker hooks as membership it already tracks.
#[async_trait]
pub trait KvEventIngress: Send + Sync {
    /// The core hands the ingress the index registry it feeds and the cancellation
    /// scope its tasks live in, once, before any other call. Ingresses that run
    /// nothing ignore it.
    fn attach(&self, index: &Arc<IndexRegistry>, cancel_token: CancellationToken) {
        let _ = (index, cancel_token);
    }

    fn open(&self, index: &IndexRegistry, key: &RoutingPartitionId, block_size: u32) -> Indexer;

    fn missing_metadata(&self, record: &WorkerCatalogRecord) -> Vec<String> {
        let _ = record;
        Vec::new()
    }

    async fn reconcile(
        &self,
        previous: Option<&WorkerCatalogRecord>,
        record: &WorkerCatalogRecord,
    ) -> Result<(), SelectionError> {
        let _ = (previous, record);
        Ok(())
    }

    async fn detach(&self, record: &WorkerCatalogRecord) {
        let _ = record;
    }
}

/// The kernel's own ingress: it opens a partition's index and nothing else. No
/// transport feeds the index; a host that listens for worker KV events
/// installs its ingress on `HostCache::index` instead.
#[derive(Debug, Default, Clone, Copy)]
pub struct LocalIndexIngress;

impl KvEventIngress for LocalIndexIngress {
    fn open(&self, index: &IndexRegistry, key: &RoutingPartitionId, block_size: u32) -> Indexer {
        index.get_or_create_indexer(key.clone(), block_size)
    }
}
