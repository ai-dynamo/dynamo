// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint for flushing the KV prefix cache across workers.
//!
//! Operators occasionally need to drop the reused KV prefix cache on workers
//! (for example after swapping weights or between benchmark runs). Each worker
//! already exposes a `clear_kv_blocks` control endpoint; this route fans that
//! control out from the frontend so a single request reaches every discovered
//! worker instance.
//!
//! ## Endpoint
//!
//! ### POST /reset_prefix_cache
//!
//! Broadcasts the `clear_kv_blocks` control to registered worker instances and
//! reports, per instance, whether the flush succeeded.
//!
//! An optional JSON body scopes the flush to a single namespace:
//!
//! ```json
//! // Request (optional) — omit the body to flush every discovered namespace
//! {"namespace": "my-namespace"}
//!
//! // Response
//! {
//!   "cleared_workers": [
//!     {"name": "dynamo/backend-instance-123", "endpoint": "dynamo/backend/clear_kv_blocks", "status": "Successfully cleared kv blocks for instance"}
//!   ],
//!   "failed_workers": []
//! }
//! ```
//!
//! In a shared discovery backend (one etcd serving several namespaces) an
//! unscoped request flushes every namespace it can see; pass `namespace` to
//! restrict the blast radius to a single tenant.
//!
//! This route is part of the frontend admin API and is only registered when the
//! admin API is enabled (see `DYN_DISABLE_FRONTEND_ADMIN_API`).

use super::{RouteDoc, service_v2};
use axum::{
    Json, Router,
    extract::State,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use dynamo_runtime::{
    DistributedRuntime, discovery::DiscoveryInstance, discovery::DiscoveryQuery,
    pipeline::PushRouter, protocols::annotated::Annotated, stream::StreamExt,
};

/// Worker-side control endpoint that flushes the KV block / prefix cache.
pub const CLEAR_KV_ENDPOINT: &str = "clear_kv_blocks";

type ClearKvRouter = PushRouter<(), Annotated<serde_json::Value>>;

/// Cache of PushRouters keyed by `(namespace, component)`.
///
/// Building a router calls `endpoint.client()`, which spawns a background
/// monitor task bound to the runtime's cancellation token. Without caching, a
/// fresh task would leak on every request; memoizing bounds the tasks to the
/// number of worker groups.
type RouterCache = Arc<Mutex<HashMap<(String, String), Arc<ClearKvRouter>>>>;

/// Combined router state: the HTTP service state (for discovery), the
/// distributed runtime used to issue RPCs to workers, and a cache of per-group
/// routers.
#[derive(Clone)]
struct ResetPrefixCacheState {
    service: Arc<service_v2::State>,
    drt: Option<Arc<DistributedRuntime>>,
    router_cache: RouterCache,
}

/// Optional request body used to scope the flush.
#[derive(Debug, Default, Deserialize)]
struct ResetPrefixCacheRequest {
    /// When set, only worker groups in this namespace are flushed.
    #[serde(default)]
    namespace: Option<String>,
}

/// Build the `/reset_prefix_cache` admin route.
///
/// `drt` is the runtime used to reach worker components; when `None` the route
/// still registers but returns `503 Service Unavailable` so the endpoint's
/// absence is never silently a `404`.
pub fn reset_prefix_cache_router(
    service: Arc<service_v2::State>,
    drt: Option<Arc<DistributedRuntime>>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/reset_prefix_cache".to_string());

    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::POST, &path)];

    let state = ResetPrefixCacheState {
        service,
        drt,
        router_cache: Arc::new(Mutex::new(HashMap::new())),
    };

    let router = Router::new()
        .route(&path, post(reset_prefix_cache_handler))
        .with_state(state);

    (docs, router)
}

/// Return a cached PushRouter for `(namespace, component)`, creating and caching
/// one on first use so the underlying client monitor task is not re-spawned per
/// request.
async fn get_or_create_router(
    state: &ResetPrefixCacheState,
    drt: &DistributedRuntime,
    namespace: &str,
    component: &str,
) -> anyhow::Result<Arc<ClearKvRouter>> {
    let key = (namespace.to_string(), component.to_string());

    {
        let cache = state.router_cache.lock().await;
        if let Some(router) = cache.get(&key) {
            return Ok(router.clone());
        }
    }

    let component_obj = drt.namespace(namespace)?.component(component)?;
    let endpoint = component_obj.endpoint(CLEAR_KV_ENDPOINT);
    let client = endpoint.client().await?;
    let router = Arc::new(ClearKvRouter::from_client(client, Default::default()).await?);

    let mut cache = state.router_cache.lock().await;
    // Another request may have inserted while we built ours; prefer the existing
    // entry so callers share a single router (and monitor task) per group.
    let router = cache.entry(key).or_insert(router).clone();
    Ok(router)
}

async fn reset_prefix_cache_handler(
    State(state): State<ResetPrefixCacheState>,
    body: Option<Json<ResetPrefixCacheRequest>>,
) -> impl IntoResponse {
    let filter = body.map(|Json(b)| b).unwrap_or_default();

    let Some(drt) = state.drt.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "message": "Distributed runtime not available" })),
        );
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
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "message": format!("Failed to list endpoints: {e}") })),
            );
        }
    };

    // Collect the unique namespace/component pairs that expose clear_kv_blocks,
    // honoring the optional namespace filter.
    let mut worker_groups: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for instance in &all_instances {
        if let DiscoveryInstance::Endpoint(inst) = instance
            && inst.endpoint == CLEAR_KV_ENDPOINT
        {
            if let Some(ns) = filter.namespace.as_deref()
                && inst.namespace != ns
            {
                continue;
            }
            let key = (inst.namespace.clone(), inst.component.clone());
            if seen.insert(key.clone()) {
                worker_groups.push(key);
            }
        }
    }

    if worker_groups.is_empty() {
        let message = match filter.namespace.as_deref() {
            Some(ns) => format!("No active worker groups found in namespace '{ns}'"),
            None => "No active worker groups found".to_string(),
        };
        return (StatusCode::NOT_FOUND, Json(json!({ "message": message })));
    }

    let mut cleared_workers = Vec::new();
    let mut failed_workers = Vec::new();

    for (namespace, component) in &worker_groups {
        tracing::debug!("Processing worker group: {}/{}", namespace, component);

        let endpoint_label = format!("{namespace}/{component}/{CLEAR_KV_ENDPOINT}");

        let router = match get_or_create_router(&state, &drt, namespace, component).await {
            Ok(router) => router,
            Err(e) => {
                failed_workers.push(json!({
                    "name": format!("{namespace}/{component}"),
                    "endpoint": endpoint_label,
                    "status": "Failed to connect to worker group",
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
                    Some(response) if response.is_error() => {
                        let error = response
                            .error
                            .as_ref()
                            .map(|e| e.to_string())
                            .or_else(|| response.comment.as_ref().map(|c| c.join(", ")))
                            .unwrap_or_else(|| "worker reported an error".to_string());
                        failed_workers.push(json!({
                            "name": instance_name,
                            "endpoint": endpoint_label,
                            "status": "Worker reported an error clearing kv blocks",
                            "error": error,
                        }));
                    }
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

    // Report an error status when nothing could be cleared but failures occurred;
    // otherwise 200 with the per-instance breakdown (which may include partial
    // failures in `failed_workers`).
    let status = if cleared_workers.is_empty() && !failed_workers.is_empty() {
        StatusCode::INTERNAL_SERVER_ERROR
    } else {
        StatusCode::OK
    };

    (
        status,
        Json(json!({
            "cleared_workers": cleared_workers,
            "failed_workers": failed_workers,
        })),
    )
}
