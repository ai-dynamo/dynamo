// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use dynamo_kv_router::protocols::{LocalBlockHash, WorkerId, compute_block_hash_for_seq};

use super::registry::WorkerRegistry;

pub struct AppState {
    pub registry: WorkerRegistry,
    pub block_size: u32,
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub instance_id: WorkerId,
    pub endpoint: String,
    #[serde(default)]
    pub dp_rank: Option<u32>,
}

#[derive(Deserialize)]
pub struct UnregisterRequest {
    pub instance_id: WorkerId,
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
    #[serde(default)]
    pub lora_name: Option<String>,
}

#[derive(Deserialize)]
pub struct QueryByHashRequest {
    pub block_hashes: Vec<i64>,
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
        .register(req.instance_id, req.endpoint, req.dp_rank.unwrap_or(0))
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
    let result = match req.dp_rank {
        Some(dp_rank) => {
            state
                .registry
                .deregister_dp_rank(req.instance_id, dp_rank)
                .await
        }
        None => state.registry.deregister(req.instance_id).await,
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

fn build_score_response(overlap: dynamo_kv_router::protocols::OverlapScores) -> ScoreResponse {
    let mut scores: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for (k, v) in &overlap.scores {
        scores
            .entry(k.worker_id.to_string())
            .or_default()
            .insert(k.dp_rank.to_string(), *v);
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
    let block_hashes = compute_block_hash_for_seq(
        &req.token_ids,
        state.block_size,
        None,
        req.lora_name.as_deref(),
    );
    match state.registry.indexer().find_matches(block_hashes).await {
        Ok(overlap) => (
            StatusCode::OK,
            Json(serde_json::json!(build_score_response(overlap))),
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
    let block_hashes: Vec<LocalBlockHash> = req
        .block_hashes
        .iter()
        .map(|h| LocalBlockHash(*h as u64))
        .collect();
    match state.registry.indexer().find_matches(block_hashes).await {
        Ok(overlap) => (
            StatusCode::OK,
            Json(serde_json::json!(build_score_response(overlap))),
        ),
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
        .route("/register", post(register))
        .route("/unregister", post(unregister))
        .route("/workers", get(list_workers))
        .route("/query", post(query))
        .route("/query_by_hash", post(query_by_hash))
        .route("/dump", get(dump_events))
        .with_state(state)
}
