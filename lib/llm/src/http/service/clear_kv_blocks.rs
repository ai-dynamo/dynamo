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
use axum::{http::Method, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use dynamo_runtime::{
    pipeline::PushRouter, stream::StreamExt,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKvBlocksRequest {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKvBlocksResponse {
    pub success: bool,
    pub message: String,
}

pub fn clear_kv_blocks_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/clear_kv_blocks".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(clear_kv_blocks_handler))
        .with_state(state);

    (docs, router)
}

async fn clear_kv_blocks_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    tracing::trace!("Received clear all KV blocks request");

    let model_entries = state.manager().get_model_entries();

    tracing::debug!("Found {} model entries:", model_entries.len());
    for entry in &model_entries {
        tracing::debug!(
            "Entry: name={}, namespace={}, component={}",
            entry.name,
            entry.endpoint.namespace,
            entry.endpoint.component
        );
    }

    // if there are no active workers
    if model_entries.is_empty() {
        return Json(serde_json::json!({
            "message": "No active worker groups found"
        }));
    }

    let mut cleared_workers = Vec::new();
    let mut failed_workers = Vec::new();

    let distributed = match state.runtime() {
        Some(runtime) => runtime,
        None => {
            return Json(serde_json::json!({
                "message": "Failed to create distributed runtime",
            }));
        }
    };

    // create client for each model entry
    for entry in &model_entries {
        let namespace = &entry.endpoint.namespace;
        let component = &entry.endpoint.component;

        tracing::debug!("Processing worker group: {}/{}", namespace, component);

        let namespace_obj = match distributed.namespace(namespace) {
            Ok(ns) => ns,
            Err(e) => {
                failed_workers.push(json!({
                    "name": entry.name,
                    "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                    "status": "failed to get namespace",
                    "error": e.to_string()
                }));
                continue;
            }
        };

        let component_obj = match namespace_obj.component(component) {
            Ok(comp) => comp,
            Err(e) => {
                failed_workers.push(json!({
                    "name": entry.name,
                    "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                    "status": "failed to get component",
                    "error": e.to_string()
                }));
                continue;
            }
        };

        let endpoint: dynamo_runtime::component::Endpoint =
            component_obj.endpoint("clear_kv_blocks");

        let client = match endpoint.client().await {
            Ok(c) => c,
            Err(e) => {
                failed_workers.push(json!({
                    "name": entry.name,
                    "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                    "status": "failed to create client",
                    "error": e.to_string()
                }));
                continue;
            }
        };

        let router = match PushRouter::<(), serde_json::Value>::from_client(
            client.clone(),
            Default::default(),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                failed_workers.push(json!({
                    "name": entry.name,
                    "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                    "status": "failed to create router",
                    "error": e.to_string()
                }));
                continue;
            }
        };

        let instances = match component_obj.list_instances().await {
            Ok(instances) => instances,
            Err(e) => {
                failed_workers.push(json!({
                    "name": entry.name,
                    "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                    "status": "Failed to get instances for worker group",
                    "error": e.to_string()
                }));
                continue;
            }
        };

        if instances.is_empty() {
            failed_workers.push(json!({
                "name": entry.name,
                "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                "status": "No instances found for worker group",
            }));
            continue;
        }

        let instances_filtered = instances
            .clone()
            .into_iter()
            .filter(|instance| instance.endpoint == "clear_kv_blocks")
            .collect::<Vec<_>>();

        if instances_filtered.is_empty() {
            let found_endpoints: Vec<String> = instances
                .iter()
                .map(|instance| instance.endpoint.clone())
                .collect();
            failed_workers.push(json!({
                "name": entry.name,
                "endpoint": format!("{}/{}/clear_kv_blocks", namespace, component),
                "status": format!("Worker group doesn't support clear_kv_blocks. Supported endpoints: {}", found_endpoints.join(", ")),
            }));
            continue;
        }

        for instance in &instances_filtered {
            match router.round_robin(().into()).await {
                Ok(mut stream) => {
                    // Successfully sent request, now process the response
                    match stream.next().await {
                        Some(response) => {
                            // Instance successfully cleared its KV blocks
                            cleared_workers.push(json!({
                                "name": format!("{}-instance-{}", entry.name, instance.id()),
                                "endpoint": format!("{}/{}/clear_kv_blocks", entry.endpoint.namespace, entry.endpoint.component),
                                "status": "successfully cleared kv blocks for instance",
                                "response": response.to_string()
                            }));
                        }
                        None => {
                            // No response from instance
                            failed_workers.push(json!({
                                "name": format!("{}-instance-{}", entry.name, instance.id()),
                                "endpoint": format!("{}/{}/clear_kv_blocks", entry.endpoint.namespace, entry.endpoint.component),
                                "status": "no response from instance",
                            }));
                        }
                    }
                }
                Err(e) => {
                    // Failed to send request to this instance
                    failed_workers.push(json!({
                        "name": format!("{}-instance-{}", entry.name, instance.id()),
                        "endpoint": format!("{}/{}/clear_kv_blocks", entry.endpoint.namespace, entry.endpoint.component),
                        "status": "failed to send request for instance",
                        "error": e.to_string()
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({
        "message": format!("Cleared prefix cache on {} out of {} worker groups",
                          cleared_workers.len(), model_entries.len()),
        "cleared_workers": cleared_workers,
        "failed_workers": failed_workers
    }))
}
