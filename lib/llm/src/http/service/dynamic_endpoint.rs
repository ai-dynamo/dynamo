// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic endpoint handler that fans out requests to all instances that registered
//! the matching HTTP endpoint path, using the background registry.
//! Returns 404 if no instances have registered the endpoint.

use super::{RouteDoc, service_v2};
use crate::types::Annotated;
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};
use std::sync::Arc;

pub fn dynamic_endpoint_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let wildcard_path = "/{*path}";
    let path = path.unwrap_or_else(|| wildcard_path.to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(dynamic_endpoint_handler))
        .with_state(state);

    (docs, router)
}

async fn inner_dynamic_endpoint_handler(
    state: Arc<service_v2::State>,
    path: String,
    body: serde_json::Value,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    let fmt_path = format!("/{}", &path);
    let registry = state.dynamic_registry();
    let registry = match registry {
        Some(r) => r,
        None => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Dynamic registry not found",
            ));
        }
    };
    let target_clients = match registry.get_clients(&fmt_path).await {
        Some(clients) if !clients.is_empty() => clients,
        _ => return Err((StatusCode::NOT_FOUND, "Endpoint not found")),
    };

    // For now broadcast to all instances using direct routing
    let mut all_responses = Vec::new();
    for client in target_clients {
        let router = PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
            client.clone(),
            Default::default(),
        )
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Failed to get router"))?;

        let ids = client.instance_ids_avail().clone();
        for id in ids.iter() {
            let mut stream = router.direct(body.clone().into(), *id).await.map_err(|e| {
                tracing::error!("Failed to route (direct): {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Failed to route")
            })?;
            while let Some(resp) = stream.next().await {
                all_responses.push(resp);
            }
        }
    }

    Ok(Json(serde_json::json!({
        "responses": all_responses
    })))
}

async fn dynamic_endpoint_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    axum::extract::Path(path): axum::extract::Path<String>,
    body: Option<Json<serde_json::Value>>,
) -> impl IntoResponse {
    let body = body.map(|Json(v)| v).unwrap_or(serde_json::json!({}));
    inner_dynamic_endpoint_handler(state, path, body)
        .await
        .map_err(|(status_code, err_string)| {
            (
                status_code,
                Json(serde_json::json!({
                    "message": err_string
                })),
            )
        })
}
