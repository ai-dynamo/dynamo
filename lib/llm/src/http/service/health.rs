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
use serde_json::json;
use std::sync::Arc;

pub fn health_check_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/health".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &path)];

    let router = Router::new()
        .route(&path, get(health_handler))
        .with_state(state);

    (docs, router)
}

async fn health_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let model_entries = state.manager().get_model_entries();

    if model_entries.is_empty() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "message": "No endpoints available"
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
                "endpoints": endpoints
            })),
        )
    }
}
