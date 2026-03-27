// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{service_v2, RouteDoc};
use axum::{http::Method, response::IntoResponse, routing::post, Json, Router};
use serde_json::json;
use std::sync::Arc;

use dynamo_runtime::{
    discovery::DiscoveryInstance, discovery::DiscoveryQuery, pipeline::PushRouter,
    stream::StreamExt,
};

pub const RESET_PREFIX_CACHE_ENDPOINT: &str = "reset_prefix_cache";

pub fn reset_prefix_cache_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/reset_prefix_cache".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(reset_prefix_cache_handler))
        .with_state(state);

    (docs, router)
}

async fn reset_prefix_cache_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    let drt = match state.drt() {
        Some(drt) => drt,
        None => {
            return Json(serde_json::json!({
                "message": "Distributed runtime not available"
            }));
        }
    };

    // Discover all registered reset_prefix_cache endpoint instances
    let all_instances = match state.discovery().list(DiscoveryQuery::AllEndpoints).await {
        Ok(instances) => instances,
        Err(e) => {
            return Json(serde_json::json!({
                "message": format!("Failed to list endpoints: {e}")
            }));
        }
    };

    // Filter to reset_prefix_cache endpoint instances and collect unique namespace/component pairs
    let mut worker_groups: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for instance in &all_instances {
        if let DiscoveryInstance::Endpoint(inst) = instance {
            if inst.endpoint == RESET_PREFIX_CACHE_ENDPOINT {
                let key = (inst.namespace.clone(), inst.component.clone());
                if seen.insert(key.clone()) {
                    worker_groups.push(key);
                }
            }
        }
    }

    if worker_groups.is_empty() {
        return Json(serde_json::json!({
            "message": "No active worker groups found"
        }));
    }

    let mut cleared_workers = Vec::new();
    let mut failed_workers = Vec::new();

    for (namespace, component) in &worker_groups {
        tracing::debug!("Processing worker group: {}/{}", namespace, component);

        let namespace_obj = match drt.namespace(namespace) {
            Ok(ns) => ns,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{}/{}", namespace, component),
                    "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                    "status": "Failed to get namespace",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let component_obj = match namespace_obj.component(component) {
            Ok(comp) => comp,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{}/{}", namespace, component),
                    "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                    "status": "Failed to get component",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let endpoint = component_obj.endpoint(RESET_PREFIX_CACHE_ENDPOINT);

        let client = match endpoint.client().await {
            Ok(c) => c,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{}/{}", namespace, component),
                    "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                    "status": "Failed to get client",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let router = match PushRouter::<(), serde_json::Value>::from_client(
            client,
            Default::default(),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                failed_workers.push(json!({
                        "name": format!("{}/{}", namespace, component),
                        "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                        "status": "Failed to create router",
                        "error": e.to_string(),
                    }));
                continue;
            }
        };

        let discovery_key = DiscoveryQuery::Endpoint {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: RESET_PREFIX_CACHE_ENDPOINT.to_string(),
        };

        let discovery_instances = match state.discovery().list(discovery_key).await {
            Ok(instances) => instances,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{}/{}", namespace, component),
                    "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                    "status": "Failed to get instances for worker group",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        if discovery_instances.is_empty() {
            failed_workers.push(json!({
                "name": format!("{}/{}", namespace, component),
                "endpoint": format!("{}/{}/{}", namespace, component, RESET_PREFIX_CACHE_ENDPOINT),
                "status": "No instances found for reset_prefix_cache endpoint",
            }));
            continue;
        }

        let instances_filtered: Vec<dynamo_runtime::component::Instance> = discovery_instances
            .into_iter()
            .filter_map(|di| match di {
                DiscoveryInstance::Endpoint(instance) => Some(instance),
                _ => None,
            })
            .collect();

        for instance in &instances_filtered {
            let instance_name = format!("{}/{}-instance-{}", namespace, component, instance.id());
            let endpoint_path = format!(
                "{}/{}/{}",
                namespace, component, RESET_PREFIX_CACHE_ENDPOINT
            );
            match router.direct(().into(), instance.id()).await {
                Ok(mut stream) => match stream.next().await {
                    Some(response) => {
                        cleared_workers.push(json!({
                            "name": instance_name,
                            "endpoint": endpoint_path,
                            "status": "Successfully reset prefix cache for instance",
                            "response": response.to_string(),
                        }));
                    }
                    None => {
                        failed_workers.push(json!({
                            "name": instance_name,
                            "endpoint": endpoint_path,
                            "status": "No response from instance",
                        }));
                    }
                },
                Err(e) => {
                    failed_workers.push(json!({
                        "name": instance_name,
                        "endpoint": endpoint_path,
                        "status": "Failed to send request for instance",
                        "error": e.to_string(),
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({
        "cleared_workers": cleared_workers,
        "failed_workers": failed_workers
    }))
}
