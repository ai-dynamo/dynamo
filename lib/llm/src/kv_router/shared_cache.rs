// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete client implementations for the [`SharedKvCache`] trait.
//!
//! Two transports are supported:
//! - **Request plane** (`SharedKvCacheRequestPlaneClient`): uses the Dynamo request plane,
//!   similar to the remote indexer pattern.
//! - **HTTP** (`SharedKvCacheHttpClient`): uses HTTP POST to a standalone service.
//!
//! Both use the same wire format:
//! - Request: `{ "block_hashes": [u64, ...] }`
//! - Response: `{ "ranges": [[start, end], ...] }` (half-open `[start, end)` ranges)
//!
//! The server-side implementation is out of scope; only the interface and client stubs
//! are defined here.

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use dynamo_kv_router::{
    SharedKvCache,
    indexer::KvRouterError,
    protocols::{LocalBlockHash, SharedCacheHits},
};
use dynamo_runtime::{
    component::Component,
    pipeline::{ManyOut, RouterMode, SingleIn, network::egress::push_router::PushRouter},
};

// ---------------------------------------------------------------------------
// Wire protocol types
// ---------------------------------------------------------------------------

/// Endpoint name for the shared KV cache query service (request plane).
pub const SHARED_KV_CACHE_QUERY_ENDPOINT: &str = "shared_kv_cache_query";

/// Request to check which blocks exist in the shared cache.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedCacheQueryRequest {
    pub block_hashes: Vec<u64>,
}

/// Response from the shared cache: sorted non-overlapping half-open ranges.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedCacheQueryResponse {
    /// Each entry is `[start, end)`.
    pub ranges: Vec<[u32; 2]>,
}

impl SharedCacheQueryResponse {
    pub fn into_shared_cache_hits(self) -> SharedCacheHits {
        let ranges: Vec<std::ops::Range<u32>> =
            self.ranges.into_iter().map(|[s, e]| s..e).collect();
        SharedCacheHits::from_ranges(ranges)
    }
}

// dynamo-llm always has dynamo-runtime, so this is unconditional.
impl dynamo_runtime::protocols::maybe_error::MaybeError for SharedCacheQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        tracing::warn!("SharedCacheQueryResponse::from_err: {err}");
        Self { ranges: vec![] }
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        None
    }
}

// ---------------------------------------------------------------------------
// Request plane client
// ---------------------------------------------------------------------------

/// Shared KV cache client using the Dynamo request plane.
pub struct SharedKvCacheRequestPlaneClient {
    router: PushRouter<SharedCacheQueryRequest, SharedCacheQueryResponse>,
}

impl SharedKvCacheRequestPlaneClient {
    pub async fn new(
        component: &Component,
        shared_cache_component_name: &str,
    ) -> anyhow::Result<Self> {
        let ns = component.namespace();
        let shared_component = ns.component(shared_cache_component_name)?;
        let endpoint = shared_component.endpoint(SHARED_KV_CACHE_QUERY_ENDPOINT);
        let client = endpoint.client().await?;
        let router =
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?;
        Ok(Self { router })
    }
}

#[async_trait]
impl SharedKvCache for SharedKvCacheRequestPlaneClient {
    async fn check_blocks(
        &self,
        block_hashes: &[LocalBlockHash],
    ) -> Result<SharedCacheHits, KvRouterError> {
        let request = SharedCacheQueryRequest {
            block_hashes: block_hashes.iter().map(|h| h.0).collect(),
        };
        let mut stream: ManyOut<SharedCacheQueryResponse> = self
            .router
            .round_robin(SingleIn::new(request))
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "Shared cache request plane query failed");
                KvRouterError::IndexerOffline
            })?;

        match stream.next().await {
            Some(resp) => Ok(resp.into_shared_cache_hits()),
            None => {
                tracing::warn!("Shared cache returned empty response");
                Ok(SharedCacheHits::default())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

/// Shared KV cache client using HTTP POST.
pub struct SharedKvCacheHttpClient {
    client: reqwest::Client,
    url: String,
}

impl SharedKvCacheHttpClient {
    pub fn new(url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
        }
    }
}

#[async_trait]
impl SharedKvCache for SharedKvCacheHttpClient {
    async fn check_blocks(
        &self,
        block_hashes: &[LocalBlockHash],
    ) -> Result<SharedCacheHits, KvRouterError> {
        let request = SharedCacheQueryRequest {
            block_hashes: block_hashes.iter().map(|h| h.0).collect(),
        };

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                tracing::warn!(error = %e, "Shared cache HTTP query failed");
                KvRouterError::IndexerOffline
            })?;

        let response = response.error_for_status().map_err(|e| {
            tracing::warn!(error = %e, "Shared cache HTTP query returned non-success status");
            KvRouterError::IndexerOffline
        })?;

        let resp: SharedCacheQueryResponse = response.json().await.map_err(|e| {
            tracing::warn!(error = %e, "Failed to parse shared cache HTTP response");
            KvRouterError::IndexerOffline
        })?;

        Ok(resp.into_shared_cache_hits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{AsyncEngine, AsyncEngineContextProvider, ResponseStream, network::Ingress},
    };
    use mockito::Server;

    struct TestQueryEngine {
        ranges: Vec<[u32; 2]>,
    }

    #[dynamo_runtime::pipeline::async_trait]
    impl
        AsyncEngine<
            SingleIn<SharedCacheQueryRequest>,
            ManyOut<SharedCacheQueryResponse>,
            anyhow::Error,
        > for TestQueryEngine
    {
        async fn generate(
            &self,
            request: SingleIn<SharedCacheQueryRequest>,
        ) -> anyhow::Result<ManyOut<SharedCacheQueryResponse>> {
            let (_request, ctx) = request.into_parts();
            let response = SharedCacheQueryResponse {
                ranges: self.ranges.clone(),
            };
            Ok(ResponseStream::new(
                Box::pin(dynamo_runtime::stream::iter(vec![response])),
                ctx.context(),
            ))
        }
    }

    async fn make_test_runtime() -> DistributedRuntime {
        let runtime = Runtime::from_current().unwrap();
        DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_shared_kv_cache_http_client_success() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock("POST", "/check_blocks")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"ranges":[[0,2],[4,5]]}"#)
            .create_async()
            .await;

        let client = SharedKvCacheHttpClient::new(format!("{}/check_blocks", server.url()));
        let hits = client
            .check_blocks(&[
                LocalBlockHash(11),
                LocalBlockHash(22),
                LocalBlockHash(33),
                LocalBlockHash(44),
                LocalBlockHash(55),
            ])
            .await
            .unwrap();

        assert_eq!(hits.ranges, vec![0..2, 4..5]);
        assert_eq!(hits.total_hits, 3);
    }

    #[tokio::test]
    async fn test_shared_kv_cache_http_client_rejects_non_success_status() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock("POST", "/check_blocks")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(r#"{"ranges":[[0,2]]}"#)
            .create_async()
            .await;

        let client = SharedKvCacheHttpClient::new(format!("{}/check_blocks", server.url()));
        let result = client
            .check_blocks(&[LocalBlockHash(11), LocalBlockHash(22)])
            .await;

        assert!(matches!(result, Err(KvRouterError::IndexerOffline)));
    }

    #[tokio::test]
    async fn test_shared_kv_cache_http_client_rejects_malformed_json() {
        let mut server = Server::new_async().await;
        let _mock = server
            .mock("POST", "/check_blocks")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"not_ranges":[[0,2]]}"#)
            .create_async()
            .await;

        let client = SharedKvCacheHttpClient::new(format!("{}/check_blocks", server.url()));
        let result = client
            .check_blocks(&[LocalBlockHash(11), LocalBlockHash(22)])
            .await;

        assert!(matches!(result, Err(KvRouterError::IndexerOffline)));
    }

    #[tokio::test]
    async fn test_shared_kv_cache_request_plane_client_success() {
        let drt = make_test_runtime().await;
        let namespace = drt.namespace("test-ns-shared-cache").unwrap();
        let shared_cache_component = namespace.component("shared-cache").unwrap();
        let endpoint = shared_cache_component.endpoint(SHARED_KV_CACHE_QUERY_ENDPOINT);

        let ingress = Ingress::<SingleIn<SharedCacheQueryRequest>, ManyOut<SharedCacheQueryResponse>>::for_engine(
            Arc::new(TestQueryEngine {
                ranges: vec![[1, 3], [4, 5]],
            }),
        )
        .unwrap();

        let server_task = tokio::spawn(
            endpoint
                .endpoint_builder()
                .handler(ingress)
                .graceful_shutdown(true)
                .start(),
        );

        let endpoint_client = endpoint.client().await.unwrap();
        endpoint_client.wait_for_instances().await.unwrap();

        let caller_component = namespace.component("router").unwrap();
        let client = SharedKvCacheRequestPlaneClient::new(&caller_component, "shared-cache")
            .await
            .unwrap();

        let hits = client
            .check_blocks(&[
                LocalBlockHash(10),
                LocalBlockHash(20),
                LocalBlockHash(30),
                LocalBlockHash(40),
                LocalBlockHash(50),
            ])
            .await
            .unwrap();

        assert_eq!(hits.ranges, vec![1..3, 4..5]);
        assert_eq!(hits.total_hits, 3);

        server_task.abort();
    }
}
