// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo RL worker discovery surface.
//!
//! Workers that run with `DYN_ENABLE_RL` / `--enable-rl` register an `rl`
//! request-plane endpoint:
//!
//! ```text
//! dyn://<namespace>.<component>.rl
//! ```
//!
//! This crate exposes a read-only frontend route. The frontend discovers live
//! `rl` endpoint instances from Dynamo discovery, then asks each worker for its
//! available RL admin routes with `{"method": "routes"}` over the request
//! plane. It does not expose a frontend fan-out method endpoint.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::{
    DistributedRuntime,
    component::{Client, Instance, TransportType},
    discovery::{DiscoveryInstance, DiscoveryQuery},
    pipeline::{
        SingleIn,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};
use futures::{StreamExt, future::join_all};

const DEFAULT_NAMESPACE: &str = "dynamo";
const DEFAULT_RL_ENDPOINT: &str = "rl";
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

type ModelKey = (String, String, u64);

#[derive(Clone)]
pub struct RlDiscoveryConfig {
    pub runtime: Arc<DistributedRuntime>,
    pub namespace: String,
    pub rl_endpoint: String,
    pub component_filter: Option<Vec<String>>,
    pub request_timeout: Duration,
}

impl RlDiscoveryConfig {
    pub fn from_env(runtime: Arc<DistributedRuntime>) -> Self {
        let namespace = std::env::var("DYN_NAMESPACE").unwrap_or_else(|_| DEFAULT_NAMESPACE.into());
        let rl_endpoint =
            std::env::var("DYN_RL_ENDPOINT").unwrap_or_else(|_| DEFAULT_RL_ENDPOINT.into());
        let component_filter = parse_csv_env("DYN_RL_COMPONENTS")
            .or_else(|| std::env::var("DYN_RL_COMPONENT").ok().map(|c| vec![c]));
        let request_timeout = std::env::var("DYN_RL_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS));

        Self {
            runtime,
            namespace,
            rl_endpoint,
            component_filter,
            request_timeout,
        }
    }
}

fn parse_csv_env(name: &str) -> Option<Vec<String>> {
    let values = std::env::var(name).ok()?;
    let parsed = values
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    (!parsed.is_empty()).then_some(parsed)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RlWorkerInfo {
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub instance_id: u64,
    pub transport: TransportType,
    pub request_plane_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub routes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RlWorkersResponse {
    pub namespace: String,
    pub workers: Vec<RlWorkerInfo>,
}

#[derive(Clone)]
pub struct RlDiscoveryState {
    config: Arc<RlDiscoveryConfig>,
}

impl RlDiscoveryState {
    pub fn new(config: RlDiscoveryConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

pub fn rl_router(state: RlDiscoveryState) -> Router {
    Router::new()
        .route("/v1/rl/workers", get(workers_handler))
        .with_state(state)
}

async fn workers_handler(State(state): State<RlDiscoveryState>) -> impl IntoResponse {
    match list_workers(&state.config).await {
        Ok(workers) => Json(RlWorkersResponse {
            namespace: state.config.namespace.clone(),
            workers,
        })
        .into_response(),
        Err(err) => {
            tracing::error!("failed to list RL workers: {err}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "discovery_failed",
                    "message": err.to_string(),
                })),
            )
                .into_response()
        }
    }
}

async fn list_workers(config: &RlDiscoveryConfig) -> anyhow::Result<Vec<RlWorkerInfo>> {
    let endpoint_instances = config
        .runtime
        .discovery()
        .list(DiscoveryQuery::NamespacedEndpoints {
            namespace: config.namespace.clone(),
        })
        .await?;

    let model_instances = config
        .runtime
        .discovery()
        .list(DiscoveryQuery::NamespacedModels {
            namespace: config.namespace.clone(),
        })
        .await
        .unwrap_or_default();

    let models = model_map(model_instances);
    let rl_endpoints = endpoint_instances
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Endpoint(endpoint) => Some(endpoint),
            _ => None,
        })
        .filter(|endpoint| endpoint.endpoint == config.rl_endpoint)
        .filter(|endpoint| {
            config
                .component_filter
                .as_ref()
                .map(|components| components.iter().any(|c| c == &endpoint.component))
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();

    let mut workers = join_all(rl_endpoints.into_iter().map(|endpoint| {
        let runtime = config.runtime.clone();
        let timeout = config.request_timeout;
        let model = models
            .get(&(
                endpoint.namespace.clone(),
                endpoint.component.clone(),
                endpoint.instance_id,
            ))
            .cloned();
        async move { describe_worker(runtime, endpoint, model, timeout).await }
    }))
    .await;

    workers.sort_by(|a, b| {
        (&a.namespace, &a.component, &a.endpoint, a.instance_id).cmp(&(
            &b.namespace,
            &b.component,
            &b.endpoint,
            b.instance_id,
        ))
    });

    workers.dedup_by(|a, b| {
        a.namespace == b.namespace
            && a.component == b.component
            && a.endpoint == b.endpoint
            && a.instance_id == b.instance_id
    });

    Ok(workers)
}

async fn describe_worker(
    runtime: Arc<DistributedRuntime>,
    endpoint: Instance,
    model: Option<String>,
    timeout: Duration,
) -> RlWorkerInfo {
    match call_worker_routes(&runtime, &endpoint, timeout).await {
        Ok(routes) => worker_info(endpoint, model, routes.routes, routes.system_url, None),
        Err(err) => worker_info(endpoint, model, Vec::new(), None, Some(err.to_string())),
    }
}

#[derive(Debug, Default)]
struct WorkerRoutes {
    routes: Vec<String>,
    system_url: Option<String>,
}

async fn call_worker_routes(
    runtime: &Arc<DistributedRuntime>,
    target: &Instance,
    timeout: Duration,
) -> anyhow::Result<WorkerRoutes> {
    let endpoint = runtime
        .namespace(&target.namespace)?
        .component(&target.component)?
        .endpoint(target.endpoint.clone());

    let client = endpoint.client().await?;
    wait_for_client_targets(&client, &[target.instance_id], Duration::from_secs(5)).await;

    let router = PushRouter::<serde_json::Value, Annotated<serde_json::Value>>::from_client(
        client,
        RouterMode::Direct,
    )
    .await?;

    let request_value = serde_json::json!({
        "method": "routes",
    });
    let instance_id = target.instance_id;

    let dispatch = async {
        let request = SingleIn::new(request_value);
        let mut stream = router.direct(request, instance_id).await?;

        while let Some(chunk) = stream.next().await {
            if let Some(data) = chunk.data {
                return parse_worker_routes(data);
            }
            if let Some(err) = chunk.error {
                anyhow::bail!(err.to_string());
            }
        }

        anyhow::bail!("empty routes response from worker");
    };

    tokio::time::timeout(timeout, dispatch)
        .await
        .map_err(|_| anyhow::anyhow!("routes request timed out after {}s", timeout.as_secs()))?
}

async fn wait_for_client_targets(client: &Client, target_ids: &[u64], timeout: Duration) {
    let wait = async {
        loop {
            let ids = client.instance_ids();
            if target_ids.iter().all(|id| ids.contains(id)) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    };
    let _ = tokio::time::timeout(timeout, wait).await;
}

fn parse_worker_routes(value: serde_json::Value) -> anyhow::Result<WorkerRoutes> {
    if value
        .get("status")
        .and_then(|status| status.as_str())
        .is_some_and(|status| status == "error")
    {
        anyhow::bail!(
            "{}",
            value
                .get("message")
                .and_then(|message| message.as_str())
                .unwrap_or("worker routes request failed")
        );
    }

    let routes = value
        .get("routes")
        .and_then(|routes| routes.as_array())
        .ok_or_else(|| anyhow::anyhow!("worker routes response missing 'routes' array"))?
        .iter()
        .filter_map(|route| route.as_str().map(ToString::to_string))
        .collect::<Vec<_>>();

    let system_url = value
        .get("system_url")
        .and_then(|url| url.as_str())
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(ToString::to_string);

    Ok(WorkerRoutes { routes, system_url })
}

fn worker_info(
    endpoint: Instance,
    model: Option<String>,
    mut routes: Vec<String>,
    system_url: Option<String>,
    error: Option<String>,
) -> RlWorkerInfo {
    routes.sort();
    routes.dedup();

    RlWorkerInfo {
        request_plane_url: request_plane_url(&endpoint),
        namespace: endpoint.namespace,
        component: endpoint.component,
        endpoint: endpoint.endpoint,
        instance_id: endpoint.instance_id,
        transport: endpoint.transport,
        system_url,
        model,
        routes,
        error,
    }
}

fn request_plane_url(endpoint: &Instance) -> String {
    format!(
        "dyn://{}.{}.{}",
        endpoint.namespace, endpoint.component, endpoint.endpoint
    )
}

fn model_map(instances: Vec<DiscoveryInstance>) -> HashMap<ModelKey, String> {
    instances
        .into_iter()
        .filter_map(|instance| match instance {
            DiscoveryInstance::Model {
                namespace,
                component,
                endpoint: _,
                instance_id,
                card_json,
                model_suffix,
            } if model_suffix.as_ref().is_none_or(|suffix| suffix.is_empty()) => {
                let model = card_json
                    .get("display_name")
                    .and_then(|value| value.as_str())
                    .map(ToString::to_string);
                model.map(|name| ((namespace, component, instance_id), name))
            }
            _ => None,
        })
        .collect()
}
