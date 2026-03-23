// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{service_v2, RouteDoc};
use axum::{http::Method, response::IntoResponse, routing::post, Json, Router};
use serde_json::json;
use std::sync::Arc;

use dynamo_runtime::{discovery::DiscoveryQuery, pipeline::PushRouter, stream::StreamExt};

pub const CLEAR_KV_ENDPOINT: &str = "clear_kv_blocks";

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
            "endpoint": format!("{}/{}/{}", ns, comp, CLEAR_KV_ENDPOINT),
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
        let namespace = &entry.endpoint_id.namespace;
        let component = &entry.endpoint_id.component;
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
            component_obj.endpoint(CLEAR_KV_ENDPOINT);

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

        let discovery_client = distributed.discovery();
        let discovery_key = DiscoveryQuery::Endpoint {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: CLEAR_KV_ENDPOINT.to_string(),
        };

        let discovery_instances = match discovery_client.list(discovery_key).await {
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

        if discovery_instances.is_empty() {
            add_worker_result(
                false,
                entry_name,
                "No instances found for clear_kv_blocks endpoint",
                namespace,
                component,
                None,
            );
            continue;
        }

        let instances_filtered: Vec<dynamo_runtime::component::Instance> = discovery_instances
            .into_iter()
            .filter_map(|di| match di {
                dynamo_runtime::discovery::DiscoveryInstance::Endpoint(instance) => Some(instance),
                _ => None,
            })
            .collect();

        for instance in &instances_filtered {
            let instance_name = format!("{}-instance-{}", entry.name, instance.id());
            match router.direct(().into(), instance.id()).await {
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
