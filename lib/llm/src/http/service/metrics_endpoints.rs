// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;

use super::{RouteDoc, service_v2};

/// List all metrics endpoints discovered in the system
async fn list_metrics_endpoints(State(state): State<Arc<service_v2::State>>) -> Response {
    // Query discovery for all metrics endpoints
    let discovery = state.discovery();
    let metrics_endpoints = match discovery
        .list(dynamo_runtime::discovery::DiscoveryQuery::AllMetricsEndpoints)
        .await
    {
        Ok(endpoints) => endpoints,
        Err(e) => {
            tracing::error!("Failed to list metrics endpoints: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to list metrics endpoints",
            )
                .into_response();
        }
    };

    // Transform discovery instances to response format
    let mut endpoints = Vec::new();
    for instance in metrics_endpoints {
        if let dynamo_runtime::discovery::DiscoveryInstance::MetricsEndpoint {
            namespace,
            instance_id,
            url,
            ..
        } = instance
        {
            endpoints.push(MetricsEndpointListing {
                namespace,
                instance_id: format!("{:x}", instance_id),
                url,
            });
        }
    }

    let response = ListMetricsEndpoints {
        object: "list",
        data: endpoints,
    };

    Json(response).into_response()
}

#[derive(Serialize)]
struct ListMetricsEndpoints {
    object: &'static str,
    data: Vec<MetricsEndpointListing>,
}

#[derive(Serialize)]
struct MetricsEndpointListing {
    namespace: String,
    instance_id: String,
    url: String,
}

/// Create an Axum [`Router`] for listing metrics endpoints
/// If no path is provided, the default path is `/v1/metrics_endpoints`
pub fn list_metrics_endpoints_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/metrics_endpoints".to_string());
    let doc = RouteDoc::new(axum::http::Method::GET, &path);

    let router = Router::new()
        .route(&path, get(list_metrics_endpoints))
        .with_state(state);

    (vec![doc], router)
}
