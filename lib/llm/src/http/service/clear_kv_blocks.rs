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
use serde_json::json;
use std::sync::Arc;

use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};

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
    let model_entries = state.manager().get_model_entries();

    // if there are no active workers
    if model_entries.is_empty() {
        return Json(serde_json::json!({
            "message": "No active worker groups found"
        }));
    }

    let distributed = match state.runtime() {
        Some(runtime) => runtime,
        None => {
            return Json(serde_json::json!({
                "message": "Failed to create distributed runtime",
            }));
        }
    };

    let mut cleared_workers = Vec::new();
    let mut failed_workers = Vec::new();

    // update cleared and failed workers
    let mut add_worker_result = |success: bool,
                                 name: String,
                                 status: &str,
                                 ns: &str,
                                 comp: &str,
                                 message: Option<String>| {
        let mut result = json!({
            "name": name,
            "endpoint": format!("{}/{}/clear_kv_blocks", ns, comp),
            "status": status,
        });
        if success {
            if let Some(m) = message {
                result["response"] = json!(m);
            }
            cleared_workers.push(result);
        } else {
            if let Some(m) = message {
                result["error"] = json!(m);
            }
            failed_workers.push(result);
        }
    };

    // create client for each model entry
    for entry in &model_entries {
        let namespace = &entry.endpoint.namespace;
        let component = &entry.endpoint.component;
        let entry_name = entry.name.to_string();

        tracing::debug!("Processing worker group: {}/{}", namespace, component);

        let namespace_obj = match distributed.namespace(namespace) {
            Ok(ns) => ns,
            Err(e) => {
                add_worker_result(
                    false,
                    entry_name,
                    "Failed to get namespace",
                    namespace,
                    component,
                    Some(e.to_string()),
                );
                continue;
            }
        };

        let component_obj = match namespace_obj.component(component) {
            Ok(comp) => comp,
            Err(e) => {
                add_worker_result(
                    false,
                    entry_name,
                    "Failed to get component",
                    namespace,
                    component,
                    Some(e.to_string()),
                );
                continue;
            }
        };

        let endpoint: dynamo_runtime::component::Endpoint =
            component_obj.endpoint("clear_kv_blocks");

        let client = match endpoint.client().await {
            Ok(c) => c,
            Err(e) => {
                add_worker_result(
                    false,
                    entry_name,
                    "Failed to get client",
                    namespace,
                    component,
                    Some(e.to_string()),
                );
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
                add_worker_result(
                    false,
                    entry_name,
                    "Failed to create router",
                    namespace,
                    component,
                    Some(e.to_string()),
                );
                continue;
            }
        };

        let instances = match component_obj.list_instances().await {
            Ok(instances) => instances,
            Err(e) => {
                add_worker_result(
                    false,
                    entry_name,
                    "Failed to get instances for worker group",
                    namespace,
                    component,
                    Some(e.to_string()),
                );
                continue;
            }
        };

        if instances.is_empty() {
            add_worker_result(
                false,
                entry_name,
                "No instances found for worker group",
                namespace,
                component,
                None,
            );
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
            add_worker_result(
                false,
                entry_name,
                &format!(
                    "Worker group doesn't support clear_kv_blocks. Supported endpoints: {}",
                    found_endpoints.join(", ")
                ),
                namespace,
                component,
                None,
            );
            continue;
        }

        for instance in &instances_filtered {
            let instance_name = format!("{}-instance-{}", entry.name, instance.id());
            match router.round_robin(().into()).await {
                Ok(mut stream) => match stream.next().await {
                    Some(response) => {
                        add_worker_result(
                            true,
                            instance_name,
                            "Successfully cleared kv blocks for instance",
                            namespace,
                            component,
                            Some(response.to_string()),
                        );
                    }
                    None => {
                        add_worker_result(
                            false,
                            instance_name,
                            "No response from instance",
                            namespace,
                            component,
                            None,
                        );
                    }
                },
                Err(e) => {
                    add_worker_result(
                        false,
                        instance_name,
                        "Failed to send request for instance",
                        namespace,
                        component,
                        Some(e.to_string()),
                    );
                }
            }
        }
    }

    Json(serde_json::json!({
        "cleared_workers": cleared_workers,
        "failed_workers": failed_workers
    }))
}
