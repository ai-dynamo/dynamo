// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HiCache shared KV cache client.
//!
//! Queries sglang workers via the Dynamo request plane to check which pages
//! of a token sequence exist in L3 (HiCache) shared storage.

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use dynamo_kv_router::{
    SharedKvCache,
    indexer::KvRouterError,
    protocols::SharedCacheHits,
};
use dynamo_runtime::{
    component::Component,
    pipeline::{ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
};

// ---------------------------------------------------------------------------
// Wire protocol types (router ↔ sglang worker)
// ---------------------------------------------------------------------------

/// Endpoint name registered on each sglang worker for hicache queries.
pub const HICACHE_QUERY_ENDPOINT: &str = "hicache_query";

/// Request sent to a worker's hicache query endpoint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HicacheQueryRequest {
    pub token_ids: Vec<u32>,
}

/// Inner payload from the worker's hicache query.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HicacheQueryData {
    /// Per-page existence: `exists[i]` is true if page `i` is in L3.
    pub exists: Vec<bool>,
    /// Page size in tokens (should match the router's block_size).
    pub page_size: u32,
}

/// Response from the worker's hicache query endpoint.
///
/// Python `serve_endpoint` wraps the yielded dict in an `Annotated` envelope
/// whose `data` field holds the actual payload. After the transport layer and
/// `PushRouter` strip the outer frame, we receive `{"data": {"exists": ..., "page_size": ...}}`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HicacheQueryResponse {
    pub data: Option<HicacheQueryData>,
}

// MaybeError impls required by the request plane PushRouter.
impl dynamo_runtime::protocols::maybe_error::MaybeError for HicacheQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        tracing::warn!("HicacheQueryResponse::from_err: {err}");
        Self { data: None }
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        None
    }
}

// ---------------------------------------------------------------------------
// HicacheSharedKvCache
// ---------------------------------------------------------------------------

/// Shared KV cache client that queries sglang workers for L3 (HiCache) state
/// via the Dynamo request plane.
pub struct HicacheSharedKvCache {
    router: PushRouter<HicacheQueryRequest, HicacheQueryResponse>,
}

impl HicacheSharedKvCache {
    /// Create a client that queries the `hicache_query` endpoint on the given
    /// worker component.
    pub async fn new(component: &Component, worker_component_name: &str) -> anyhow::Result<Self> {
        let ns = component.namespace();
        let worker_component = ns.component(worker_component_name)?;
        let endpoint = worker_component.endpoint(HICACHE_QUERY_ENDPOINT);
        let client = endpoint.client().await?;
        let router =
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?;
        Ok(Self { router })
    }
}

#[async_trait]
impl SharedKvCache for HicacheSharedKvCache {
    async fn check_blocks(
        &self,
        tokens: &[u32],
        _block_size: u32,
    ) -> Result<SharedCacheHits, KvRouterError> {
        let request = HicacheQueryRequest {
            token_ids: tokens.to_vec(),
        };

        let mut stream: ManyOut<HicacheQueryResponse> = self
            .router
            .round_robin(SingleIn::new(request))
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "HiCache request plane query failed");
                KvRouterError::IndexerOffline
            })?;

        match stream.next().await {
            Some(resp) => match resp.data {
                Some(data) => {
                    let hits = SharedCacheHits::from_hits(&data.exists);
                    Ok(hits)
                }
                None => {
                    tracing::warn!("HiCache query returned response with no data");
                    Ok(SharedCacheHits::default())
                }
            },
            None => {
                tracing::warn!("HiCache query returned empty response");
                Ok(SharedCacheHits::default())
            }
        }
    }
}
