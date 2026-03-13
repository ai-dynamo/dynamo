// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
#[cfg(feature = "metrics")]
use prometheus::Encoder;
use serde::{Deserialize, Serialize};

use crate::protocols::{LocalBlockHash, WorkerId, compute_block_hash_for_seq};

use super::registry::{IndexerKey, WorkerRegistry};

pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    #[cfg(feature = "metrics")]
    pub prom_registry: prometheus::Registry,
}

fn default_tenant() -> String {
    "default".to_string()
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub instance_id: WorkerId,
    pub endpoint: String,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    pub block_size: u32,
    #[serde(default)]
    pub dp_rank: Option<u32>,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
}

#[derive(Deserialize)]
pub struct UnregisterRequest {
    pub instance_id: WorkerId,
    pub model_name: String,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub dp_rank: Option<u32>,
}

#[derive(Serialize)]
struct WorkerInfo {
    instance_id: WorkerId,
    endpoints: HashMap<u32, String>,
}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub token_ids: Vec<u32>,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
    #[serde(default)]
    pub lora_name: Option<String>,
}

#[derive(Deserialize)]
pub struct QueryByHashRequest {
    pub block_hashes: Vec<i64>,
    pub model_name: String,
    #[serde(default = "default_tenant")]
    pub tenant_id: String,
}

#[derive(Serialize)]
struct ScoreResponse {
    scores: HashMap<String, HashMap<String, u32>>,
    frequencies: Vec<usize>,
    tree_sizes: HashMap<String, HashMap<String, usize>>,
}

async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    match state
        .registry
        .register(
            req.instance_id,
            req.endpoint,
            req.dp_rank.unwrap_or(0),
            req.model_name,
            req.tenant_id,
            req.block_size,
            req.replay_endpoint,
        )
        .await
    {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({"status": "ok"})),
        ),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn unregister(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnregisterRequest>,
) -> impl IntoResponse {
    let result = match req.tenant_id {
        Some(tenant_id) => match req.dp_rank {
            Some(dp_rank) => {
                state
                    .registry
                    .deregister_dp_rank(req.instance_id, dp_rank, &req.model_name, &tenant_id)
                    .await
            }
            None => {
                state
                    .registry
                    .deregister(req.instance_id, &req.model_name, &tenant_id)
                    .await
            }
        },
        None => {
            state
                .registry
                .deregister_all_tenants(req.instance_id, &req.model_name)
                .await
        }
    };
    match result {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn list_workers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let workers: Vec<WorkerInfo> = state
        .registry
        .list()
        .into_iter()
        .map(|(instance_id, endpoints)| WorkerInfo {
            instance_id,
            endpoints,
        })
        .collect();
    Json(workers)
}

fn build_score_response(
    overlap: crate::protocols::OverlapScores,
    block_size: u32,
) -> ScoreResponse {
    let mut scores: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for (k, v) in &overlap.scores {
        scores
            .entry(k.worker_id.to_string())
            .or_default()
            .insert(k.dp_rank.to_string(), v * block_size);
    }
    let mut tree_sizes: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for (k, v) in &overlap.tree_sizes {
        tree_sizes
            .entry(k.worker_id.to_string())
            .or_default()
            .insert(k.dp_rank.to_string(), *v);
    }
    ScoreResponse {
        scores,
        frequencies: overlap.frequencies,
        tree_sizes,
    }
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let key = IndexerKey {
        model_name: req.model_name,
        tenant_id: req.tenant_id,
    };
    let Some(ie) = state.registry.get_indexer(&key) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no indexer for model={} tenant={}", key.model_name, key.tenant_id)
            })),
        );
    };
    let block_size = ie.block_size;
    let indexer = ie.indexer.clone();
    drop(ie);

    let block_hashes =
        compute_block_hash_for_seq(&req.token_ids, block_size, None, req.lora_name.as_deref());
    match indexer.find_matches(block_hashes).await {
        Ok(overlap) => (
            StatusCode::OK,
            Json(serde_json::json!(build_score_response(overlap, block_size))),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn query_by_hash(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryByHashRequest>,
) -> impl IntoResponse {
    let key = IndexerKey {
        model_name: req.model_name,
        tenant_id: req.tenant_id,
    };
    let Some(ie) = state.registry.get_indexer(&key) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": format!("no indexer for model={} tenant={}", key.model_name, key.tenant_id)
            })),
        );
    };
    let block_size = ie.block_size;
    let indexer = ie.indexer.clone();
    drop(ie);

    let block_hashes: Vec<LocalBlockHash> = req
        .block_hashes
        .iter()
        .map(|h| LocalBlockHash(*h as u64))
        .collect();
    match indexer.find_matches(block_hashes).await {
        Ok(overlap) => (
            StatusCode::OK,
            Json(serde_json::json!(build_score_response(overlap, block_size))),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

#[cfg(feature = "test-endpoints")]
#[derive(Deserialize)]
struct ListenerControlRequest {
    instance_id: WorkerId,
    #[serde(default)]
    dp_rank: Option<u32>,
}

#[cfg(feature = "test-endpoints")]
async fn test_pause_listener(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListenerControlRequest>,
) -> impl IntoResponse {
    match state
        .registry
        .pause_listener(req.instance_id, req.dp_rank.unwrap_or(0))
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

#[cfg(feature = "test-endpoints")]
async fn test_resume_listener(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ListenerControlRequest>,
) -> impl IntoResponse {
    match state
        .registry
        .resume_listener(req.instance_id, req.dp_rank.unwrap_or(0))
        .await
    {
        Ok(()) => (StatusCode::OK, Json(serde_json::json!({"status": "ok"}))),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

#[derive(Deserialize)]
struct PeerRequest {
    url: String,
}

async fn register_peer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PeerRequest>,
) -> impl IntoResponse {
    state.registry.register_peer(req.url);
    (
        StatusCode::CREATED,
        Json(serde_json::json!({"status": "ok"})),
    )
}

async fn deregister_peer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PeerRequest>,
) -> impl IntoResponse {
    if state.registry.deregister_peer(&req.url) {
        (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "peer not found"})),
        )
    }
}

async fn list_peers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.registry.list_peers())
}

async fn dump_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let all = state.registry.all_indexers_with_block_size();
    let mut handles = Vec::with_capacity(all.len());

    for (key, indexer, block_size) in all {
        handles.push(tokio::spawn(async move {
            let events = indexer.dump_events().await;
            (key, events, block_size)
        }));
    }

    let mut result: HashMap<String, serde_json::Value> = HashMap::new();
    for handle in handles {
        match handle.await {
            Ok((key, Ok(events), block_size)) => {
                let map_key = format!("{}:{}", key.model_name, key.tenant_id);
                result.insert(
                    map_key,
                    serde_json::json!({
                        "block_size": block_size,
                        "events": events,
                    }),
                );
            }
            Ok((key, Err(e), _)) => {
                let map_key = format!("{}:{}", key.model_name, key.tenant_id);
                result.insert(map_key, serde_json::json!({"error": e.to_string()}));
            }
            Err(e) => {
                tracing::warn!("dump task join error: {e}");
            }
        }
    }
    (StatusCode::OK, Json(serde_json::json!(result)))
}

async fn handle_health() -> StatusCode {
    StatusCode::OK
}

#[cfg(feature = "metrics")]
async fn handle_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let mut buf = Vec::new();
    encoder
        .encode(&state.prom_registry.gather(), &mut buf)
        .unwrap();
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            prometheus::TEXT_FORMAT.to_string(),
        )],
        buf,
    )
}

pub fn create_router(state: Arc<AppState>) -> Router {
    let router = Router::new()
        .route("/register", post(register))
        .route("/unregister", post(unregister))
        .route("/workers", get(list_workers))
        .route("/query", post(query))
        .route("/query_by_hash", post(query_by_hash))
        .route("/dump", get(dump_events))
        .route("/register_peer", post(register_peer))
        .route("/deregister_peer", post(deregister_peer))
        .route("/peers", get(list_peers))
        .route("/health", get(handle_health));

    #[cfg(feature = "test-endpoints")]
    let router = router
        .route("/test/pause_listener", post(test_pause_listener))
        .route("/test/resume_listener", post(test_resume_listener));

    let router = router.with_state(state.clone());

    #[cfg(feature = "metrics")]
    let router = {
        let metrics_route = Router::new()
            .route("/metrics", get(handle_metrics))
            .with_state(state);
        router
            .layer(axum::middleware::from_fn(
                super::metrics::metrics_middleware,
            ))
            .merge(metrics_route)
    };

    router
}
