// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RouteDoc, service_v2};
use crate::types::Annotated;
use axum::{
    Json, Router,
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::post,
};
use dynamo_runtime::instances::list_all_instances;
use dynamo_runtime::{DistributedRuntime, Runtime, component::Client};
use dynamo_runtime::{pipeline::PushRouter, stream::StreamExt};
use std::sync::Arc;

pub const DYNAMIC_ENDPOINT_PATH: &str = "dynamic_endpoint";

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

async fn dynamic_endpoint_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    axum::extract::Path(path): axum::extract::Path<String>,
) -> impl IntoResponse {
    let Some(etcd_client) = state.etcd_client() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "message": "Failed to get etcd client"
            })),
        );
    };

    let instances = match list_all_instances(etcd_client).await {
        Ok(instances) => instances,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": "Failed to get instances"
                })),
            );
        }
    };

    let dynamic_endpoints = instances
        .iter()
        .filter_map(|instance| instance.http_endpoint_path.clone())
        .collect::<Vec<String>>();

    let fmt_path = format!("/{}", &path);
    if !dynamic_endpoints.contains(&fmt_path) {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "message": "Dynamic endpoint not found"
            })),
        );
    }

    let rt = match Runtime::from_current() {
        Ok(rt) => rt,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": "Failed to get runtime"
                })),
            );
        }
    };
    let drt = match DistributedRuntime::from_settings(rt).await {
        Ok(drt) => drt,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "message": "Failed to get distributed runtime"
                })),
            );
        }
    };

    // grab all instances that expose this endpoint
    let target_instances = instances
        .iter()
        .filter(|instance| instance.http_endpoint_path == Some(fmt_path.clone()))
        .collect::<Vec<_>>();

    // use pushrouter .direct to forward the request to the filtered instances sequentially
    let mut target_clients: Vec<Client> = Vec::new();
    for instance in target_instances {
        let ns = match drt.namespace(instance.namespace.clone()) {
            Ok(ns) => ns,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "message": "Failed to get namespace"
                    })),
                );
            }
        };
        let c = match ns.component(instance.component.clone()) {
            Ok(c) => c,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "message": "Failed to get component"
                    })),
                );
            }
        };
        let ep = c.endpoint(path.clone());
        let c = match ep.client().await {
            Ok(c) => c,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "message": "Failed to get client"
                    })),
                );
            }
        };
        target_clients.push(c.clone());
    }

    let mut all_responses = Vec::new();
    for client in target_clients {
        let router = match PushRouter::<(), Annotated<serde_json::Value>>::from_client(
            client,
            Default::default(),
        )
        .await
        {
            Ok(router) => router,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "message": "Failed to get router"
                    })),
                );
            }
        };
        let mut stream = match router.round_robin(().into()).await {
            Ok(s) => s,
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"message": "Failed to route"})),
                );
            }
        };

        while let Some(resp) = stream.next().await {
            all_responses.push(resp);
        }
    }

    return (
        StatusCode::OK,
        Json(serde_json::json!({
            "responses": all_responses
        })),
    );
}
