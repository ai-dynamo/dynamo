// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use dynamo_kv_router::protocols::{LocalBlockHash, WorkerId, compute_block_hash_for_seq};

use super::registry::WorkerRegistry;

pub struct AppState {
    pub registry: WorkerRegistry,
    pub block_size: u32,
}

#[derive(Deserialize)]
pub struct RegisterWorkerRequest {
    pub worker_id: WorkerId,
    pub zmq_addresses: HashMap<u32, String>,
}

#[derive(Serialize)]
struct WorkerInfo {
    worker_id: WorkerId,
    zmq_addresses: HashMap<u32, String>,
}

#[derive(Deserialize)]
pub struct ScoreRequest {
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub lora_name: Option<String>,
}

#[derive(Deserialize)]
pub struct ScoreHashedRequest {
    pub block_hashes: Vec<i64>,
}

#[derive(Serialize)]
struct ScoreResponse {
    scores: std::collections::HashMap<String, u32>,
    frequencies: Vec<usize>,
    tree_sizes: std::collections::HashMap<String, usize>,
}

async fn register_worker(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterWorkerRequest>,
) -> impl IntoResponse {
    match state.registry.register(req.worker_id, req.zmq_addresses) {
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

async fn deregister_worker(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<WorkerId>,
) -> impl IntoResponse {
    match state.registry.deregister(worker_id).await {
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
        .map(|(worker_id, zmq_addresses)| WorkerInfo {
            worker_id,
            zmq_addresses,
        })
        .collect();
    Json(workers)
}

async fn score(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScoreRequest>,
) -> impl IntoResponse {
    let block_hashes = compute_block_hash_for_seq(
        &req.tokens,
        state.block_size,
        None,
        req.lora_name.as_deref(),
    );
    match state.registry.indexer().find_matches(block_hashes).await {
        Ok(overlap) => {
            let resp = ScoreResponse {
                scores: overlap
                    .scores
                    .iter()
                    .map(|(k, v)| (format!("{}:{}", k.worker_id, k.dp_rank), *v))
                    .collect(),
                frequencies: overlap.frequencies,
                tree_sizes: overlap
                    .tree_sizes
                    .iter()
                    .map(|(k, v)| (format!("{}:{}", k.worker_id, k.dp_rank), *v))
                    .collect(),
            };
            (StatusCode::OK, Json(serde_json::json!(resp)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn score_hashed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScoreHashedRequest>,
) -> impl IntoResponse {
    let block_hashes: Vec<LocalBlockHash> = req
        .block_hashes
        .iter()
        .map(|h| LocalBlockHash(*h as u64))
        .collect();
    match state.registry.indexer().find_matches(block_hashes).await {
        Ok(overlap) => {
            let resp = ScoreResponse {
                scores: overlap
                    .scores
                    .iter()
                    .map(|(k, v)| (format!("{}:{}", k.worker_id, k.dp_rank), *v))
                    .collect(),
                frequencies: overlap.frequencies,
                tree_sizes: overlap
                    .tree_sizes
                    .iter()
                    .map(|(k, v)| (format!("{}:{}", k.worker_id, k.dp_rank), *v))
                    .collect(),
            };
            (StatusCode::OK, Json(serde_json::json!(resp)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn dump_events(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.registry.indexer().dump_events().await {
        Ok(events) => (StatusCode::OK, Json(serde_json::json!(events))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/workers", post(register_worker))
        .route("/workers", get(list_workers))
        .route("/workers/{worker_id}", delete(deregister_worker))
        .route("/score", post(score))
        .route("/score_hashed", post(score_hashed))
        .route("/dump", get(dump_events))
        .with_state(state)
}
