// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::instances::list_all_instances;
use serde_json::json;
use std::sync::Arc;

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let health_path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &health_path)];

    let router = Router::new()
        .route(&health_path, get(health_handler))
        .with_state(state);

    (docs, router)
}

pub fn live_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let live_path = path.unwrap_or_else(|| "/live".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &live_path)];

    let router = Router::new()
        .route(&live_path, get(live_handler))
        .with_state(state);

    (docs, router)
}

async fn live_handler(
    axum::extract::State(_state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "status": "live",
            "message": "Service is live"
        })),
    )
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let instances = if let Some(etcd_client) = state.etcd_client() {
        match list_all_instances(etcd_client).await {
            Ok(instances) => instances,
            Err(err) => {
                tracing::warn!("Failed to fetch instances from etcd: {}", err);
                vec![]
            }
        }
    } else {
        vec![]
    };

    let mut endpoints: Vec<String> = instances
        .iter()
        .map(|instance| instance.endpoint_id().as_url())
        .collect();
    endpoints.sort();
    endpoints.dedup();
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "endpoints": endpoints,
            "instances": instances
        })),
    )
}
