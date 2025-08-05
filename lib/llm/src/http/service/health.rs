// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{service_v2, RouteDoc};
use axum::{http::Method, http::StatusCode, response::IntoResponse, routing::get, Json, Router};
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
    let model_entries = state.manager().get_model_entries();
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

    if model_entries.is_empty() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "message": "No endpoints available",
                "instances": instances
            })),
        )
    } else {
        let endpoints: Vec<String> = model_entries
            .iter()
            .map(|entry| entry.endpoint.as_url())
            .collect();
        (
            StatusCode::OK,
            Json(json!({
                "status": "healthy",
                "endpoints": endpoints,
                "instances": instances
            })),
        )
    }
}
