// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP server and in-memory store for the dummy shared KV cache.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use dashmap::DashSet;
use serde::{Deserialize, Serialize};

use crate::protocols::SharedCacheHits;

/// Endpoint name for the shared KV cache query service (request plane).
/// Matches the constant in `lib/llm/src/kv_router/shared_cache.rs`.
pub const SHARED_KV_CACHE_QUERY_ENDPOINT: &str = "shared_kv_cache_query";

// ---------------------------------------------------------------------------
// Wire protocol types (shared with lib/llm/src/kv_router/shared_cache.rs)
// ---------------------------------------------------------------------------

/// Request to check which blocks exist in the shared cache.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedCacheQueryRequest {
    pub block_hashes: Vec<u64>,
}

/// Response: sorted non-overlapping half-open ranges of present block positions.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedCacheQueryResponse {
    pub ranges: Vec<[u32; 2]>,
}

#[cfg(feature = "indexer-runtime")]
impl crate::indexer::MaybeError for SharedCacheQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        tracing::warn!("SharedCacheQueryResponse::from_err: {err}");
        Self { ranges: vec![] }
    }
    fn err(&self) -> Option<Box<dyn std::error::Error + Send + Sync>> {
        None
    }
}

#[cfg(feature = "indexer-runtime")]
impl dynamo_runtime::protocols::maybe_error::MaybeError for SharedCacheQueryResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        tracing::warn!("SharedCacheQueryResponse::from_err: {err}");
        Self { ranges: vec![] }
    }
    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        None
    }
}

/// Request to store block hashes (for populating the dummy cache).
#[derive(Deserialize)]
pub struct StoreRequest {
    pub block_hashes: Vec<u64>,
}

/// Request to remove block hashes.
#[derive(Deserialize)]
pub struct RemoveRequest {
    pub block_hashes: Vec<u64>,
}

// ---------------------------------------------------------------------------
// In-memory store
// ---------------------------------------------------------------------------

/// Thread-safe set of block hashes that exist in the "shared cache".
pub struct SharedCacheStore {
    blocks: DashSet<u64>,
}

impl SharedCacheStore {
    pub fn new() -> Self {
        Self {
            blocks: DashSet::new(),
        }
    }

    /// Insert block hashes into the store.
    pub fn store(&self, hashes: &[u64]) {
        for &h in hashes {
            self.blocks.insert(h);
        }
    }

    /// Remove block hashes from the store.
    pub fn remove(&self, hashes: &[u64]) {
        for &h in hashes {
            self.blocks.remove(&h);
        }
    }

    /// Check which positions in the request have their block hash present.
    /// Returns coalesced ranges for the response.
    pub fn check_blocks(&self, block_hashes: &[u64]) -> SharedCacheHits {
        let hits: Vec<bool> = block_hashes
            .iter()
            .map(|h| self.blocks.contains(h))
            .collect();
        SharedCacheHits::from_hits(&hits)
    }

    /// Number of blocks currently stored.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }
}

// ---------------------------------------------------------------------------
// Axum handlers
// ---------------------------------------------------------------------------

pub struct AppState {
    pub store: Arc<SharedCacheStore>,
}

/// POST /check_blocks — query which block hashes exist.
async fn check_blocks(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SharedCacheQueryRequest>,
) -> impl IntoResponse {
    let hits = state.store.check_blocks(&req.block_hashes);
    let ranges: Vec<[u32; 2]> = hits.ranges.iter().map(|r| [r.start, r.end]).collect();
    (StatusCode::OK, Json(SharedCacheQueryResponse { ranges }))
}

/// POST /store — add block hashes to the cache.
async fn store_blocks(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StoreRequest>,
) -> impl IntoResponse {
    let count = req.block_hashes.len();
    state.store.store(&req.block_hashes);
    (
        StatusCode::CREATED,
        Json(serde_json::json!({
            "status": "ok",
            "stored": count,
            "total": state.store.len(),
        })),
    )
}

/// POST /remove — remove block hashes from the cache.
async fn remove_blocks(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RemoveRequest>,
) -> impl IntoResponse {
    let count = req.block_hashes.len();
    state.store.remove(&req.block_hashes);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "removed": count,
            "total": state.store.len(),
        })),
    )
}

/// GET /health — liveness check.
async fn health() -> StatusCode {
    StatusCode::OK
}

/// GET /stats — number of blocks stored.
async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "total_blocks": state.store.len(),
    }))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/check_blocks", post(check_blocks))
        .route("/store", post(store_blocks))
        .route("/remove", post(remove_blocks))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Request-plane engine (for Dynamo runtime integration)
// ---------------------------------------------------------------------------

#[cfg(feature = "indexer-runtime")]
pub struct SharedCacheQueryEngine {
    pub store: Arc<SharedCacheStore>,
}

#[cfg(feature = "indexer-runtime")]
#[dynamo_runtime::pipeline::async_trait]
impl
    dynamo_runtime::pipeline::AsyncEngine<
        dynamo_runtime::pipeline::SingleIn<SharedCacheQueryRequest>,
        dynamo_runtime::pipeline::ManyOut<SharedCacheQueryResponse>,
        anyhow::Error,
    > for SharedCacheQueryEngine
{
    async fn generate(
        &self,
        request: dynamo_runtime::pipeline::SingleIn<SharedCacheQueryRequest>,
    ) -> anyhow::Result<dynamo_runtime::pipeline::ManyOut<SharedCacheQueryResponse>> {
        use dynamo_runtime::pipeline::{AsyncEngineContextProvider, ResponseStream};

        let (req, ctx) = request.into_parts();
        let hits = self.store.check_blocks(&req.block_hashes);
        let ranges: Vec<[u32; 2]> = hits.ranges.iter().map(|r| [r.start, r.end]).collect();
        let response = SharedCacheQueryResponse { ranges };
        let stream = dynamo_runtime::stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_check() {
        let store = SharedCacheStore::new();
        store.store(&[100, 200, 300]);

        // Query: [100, 999, 200, 300, 888]
        // Hits at positions 0, 2, 3 => ranges [0..1, 2..4]
        let hits = store.check_blocks(&[100, 999, 200, 300, 888]);
        assert_eq!(hits.total_hits, 3);
        assert_eq!(hits.ranges, vec![0..1, 2..4]);
    }

    #[test]
    fn test_check_empty_cache() {
        let store = SharedCacheStore::new();
        let hits = store.check_blocks(&[1, 2, 3]);
        assert_eq!(hits.total_hits, 0);
        assert!(hits.ranges.is_empty());
    }

    #[test]
    fn test_remove_blocks() {
        let store = SharedCacheStore::new();
        store.store(&[10, 20, 30]);
        store.remove(&[20]);

        // Query: [10, 20, 30] => hits at 0 and 2 => ranges [0..1, 2..3]
        let hits = store.check_blocks(&[10, 20, 30]);
        assert_eq!(hits.total_hits, 2);
        assert_eq!(hits.ranges, vec![0..1, 2..3]);
    }

    #[test]
    fn test_all_hits() {
        let store = SharedCacheStore::new();
        store.store(&[1, 2, 3]);

        let hits = store.check_blocks(&[1, 2, 3]);
        assert_eq!(hits.total_hits, 3);
        assert_eq!(hits.ranges, vec![0..3]);
    }

    #[test]
    fn test_store_len() {
        let store = SharedCacheStore::new();
        assert_eq!(store.len(), 0);
        store.store(&[1, 2, 3]);
        assert_eq!(store.len(), 3);
        store.store(&[1, 4]); // 1 is a duplicate
        assert_eq!(store.len(), 4);
    }

    #[test]
    fn test_response_wire_format() {
        let hits = SharedCacheHits::from_ranges(vec![0..2, 5..8]);
        let ranges: Vec<[u32; 2]> = hits.ranges.iter().map(|r| [r.start, r.end]).collect();
        let resp = SharedCacheQueryResponse { ranges };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: SharedCacheQueryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.ranges, vec![[0, 2], [5, 8]]);
    }
}
