// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for flushing the KV prefix cache across all workers.
//!
//! Operators occasionally need to drop the reused KV prefix cache on every
//! worker (for example after swapping weights or between benchmark runs). Each
//! worker already exposes a `clear_kv_blocks` control endpoint; this route fans
//! that control out from the frontend so a single request reaches every
//! discovered worker group.
//!
//! ## Endpoint
//!
//! ### POST /reset_prefix_cache
//!
//! Broadcasts the `clear_kv_blocks` control to every registered worker instance
//! and reports, per instance, whether the flush succeeded.
//!
//! ```json
//! // Response
//! {
//!   "cleared_workers": [
//!     {"name": "dynamo/backend-instance-123", "endpoint": "dynamo/backend/clear_kv_blocks", "status": "Successfully cleared kv blocks for instance", "response": "{}"}
//!   ],
//!   "failed_workers": []
//! }
//! ```
//!
//! This route is part of the frontend admin API and is only registered when the
//! admin API is enabled (see `DYN_DISABLE_FRONTEND_ADMIN_API`) and a
//! [`DistributedRuntime`] is available to issue worker RPCs.

use super::{RouteDoc, service_v2};
use axum::{Json, Router, extract::State, http::Method, response::IntoResponse, routing::post};
use serde_json::json;
use std::sync::Arc;

use dynamo_runtime::{
    DistributedRuntime, discovery::DiscoveryInstance, discovery::DiscoveryQuery,
    pipeline::PushRouter, protocols::annotated::Annotated, stream::StreamExt,
};

/// Worker-side control endpoint that flushes the KV block / prefix cache.
pub const CLEAR_KV_ENDPOINT: &str = "clear_kv_blocks";

/// Combined router state: the HTTP service state (for discovery) plus the
/// distributed runtime used to issue RPCs to workers.
#[derive(Clone)]
struct ResetPrefixCacheState {
    service: Arc<service_v2::State>,
    drt: Option<Arc<DistributedRuntime>>,
}

/// Build the `/reset_prefix_cache` admin route.
///
/// `drt` is the runtime used to reach worker components; when `None` the route
/// still registers but returns a clear "runtime not available" message so the
/// endpoint's absence is never silently a 404.
pub fn reset_prefix_cache_router(
    service: Arc<service_v2::State>,
    drt: Option<Arc<DistributedRuntime>>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/reset_prefix_cache".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let router = Router::new()
        .route(&path, post(reset_prefix_cache_handler))
        .with_state(ResetPrefixCacheState { service, drt });

    (docs, router)
}

async fn reset_prefix_cache_handler(
    State(state): State<ResetPrefixCacheState>,
) -> impl IntoResponse {
    let Some(drt) = state.drt.as_ref() else {
        return Json(json!({
            "message": "Distributed runtime not available"
        }));
    };

    // Discover all registered clear_kv_blocks endpoint instances.
    let all_instances = match state
        .service
        .discovery()
        .list(DiscoveryQuery::AllEndpoints)
        .await
    {
        Ok(instances) => instances,
        Err(e) => {
            return Json(json!({
                "message": format!("Failed to list endpoints: {e}")
            }));
        }
    };

    // Collect the unique namespace/component pairs that expose clear_kv_blocks.
    let mut worker_groups: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for instance in &all_instances {
        if let DiscoveryInstance::Endpoint(inst) = instance
            && inst.endpoint == CLEAR_KV_ENDPOINT
        {
            let key = (inst.namespace.clone(), inst.component.clone());
            if seen.insert(key.clone()) {
                worker_groups.push(key);
            }
        }
    }

    if worker_groups.is_empty() {
        return Json(json!({
            "message": "No active worker groups found"
        }));
    }

    let mut cleared_workers = Vec::new();
    let mut failed_workers = Vec::new();

    for (namespace, component) in &worker_groups {
        tracing::debug!("Processing worker group: {}/{}", namespace, component);

        let endpoint_label = format!("{namespace}/{component}/{CLEAR_KV_ENDPOINT}");

        let component_obj = match drt
            .namespace(namespace)
            .and_then(|ns| ns.component(component))
        {
            Ok(comp) => comp,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{namespace}/{component}"),
                    "endpoint": endpoint_label,
                    "status": "Failed to resolve worker component",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let endpoint = component_obj.endpoint(CLEAR_KV_ENDPOINT);

        let client = match endpoint.client().await {
            Ok(c) => c,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{namespace}/{component}"),
                    "endpoint": endpoint_label,
                    "status": "Failed to get client",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let router = match PushRouter::<(), Annotated<serde_json::Value>>::from_client(
            client,
            Default::default(),
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{namespace}/{component}"),
                    "endpoint": endpoint_label,
                    "status": "Failed to create router",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let discovery_key = DiscoveryQuery::Endpoint {
            namespace: namespace.clone(),
            component: component.clone(),
            endpoint: CLEAR_KV_ENDPOINT.to_string(),
        };

        let discovery_instances = match state.service.discovery().list(discovery_key).await {
            Ok(instances) => instances,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{namespace}/{component}"),
                    "endpoint": endpoint_label,
                    "status": "Failed to get instances for worker group",
                    "error": e.to_string(),
                }));
                continue;
            }
        };

        let instances_filtered: Vec<dynamo_runtime::component::Instance> = discovery_instances
            .into_iter()
            .filter_map(|di| match di {
                DiscoveryInstance::Endpoint(instance) => Some(instance),
                _ => None,
            })
            .collect();

        if instances_filtered.is_empty() {
            failed_workers.push(json!({
                "name": format!("{namespace}/{component}"),
                "endpoint": endpoint_label,
                "status": "No instances found for clear_kv_blocks endpoint",
            }));
            continue;
        }

        for instance in &instances_filtered {
            let instance_name = format!("{namespace}/{component}-instance-{}", instance.id());
            match router.direct(().into(), instance.id()).await {
                Ok(mut stream) => match stream.next().await {
                    Some(response) => {
                        let response_str = response
                            .data
                            .as_ref()
                            .map(|d| d.to_string())
                            .unwrap_or_default();
                        cleared_workers.push(json!({
                            "name": instance_name,
                            "endpoint": endpoint_label,
                            "status": "Successfully cleared kv blocks for instance",
                            "response": response_str,
                        }));
                    }
                    None => {
                        failed_workers.push(json!({
                            "name": instance_name,
                            "endpoint": endpoint_label,
                            "status": "No response from instance",
                        }));
                    }
                },
                Err(e) => {
                    failed_workers.push(json!({
                        "name": instance_name,
                        "endpoint": endpoint_label,
                        "status": "Failed to send request for instance",
                        "error": e.to_string(),
                    }));
                }
            }
        }
    }

    Json(json!({
        "cleared_workers": cleared_workers,
        "failed_workers": failed_workers
    }))
}
